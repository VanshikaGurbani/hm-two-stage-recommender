"""FastAPI service: ``uvicorn hmrec.api:app``.

Environment: ``HM_SAMPLE_FRAC`` (default 0.1) selects the customer sample; ``HM_CACHE_SIZE``
(default 4096) sizes the in-process response cache.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request, Response
from pydantic import BaseModel

from . import workspace
from .images import ImageStore
from .pipeline import System


class Recommendation(BaseModel):
    article_id: int
    name: str
    product_type: str
    colour: str
    score: float
    why: str


class RecommendResponse(BaseModel):
    customer_id: str
    k: int
    lam: float
    cached: bool
    latency_ms: float
    recommendations: list[Recommendation]


class LRU:
    def __init__(self, size: int):
        self.d: OrderedDict = OrderedDict()
        self.size, self.hits, self.misses = size, 0, 0

    def get(self, key):
        if key in self.d:
            self.d.move_to_end(key)
            self.hits += 1
            return self.d[key]
        self.misses += 1
        return None

    def put(self, key, value):
        self.d[key] = value
        self.d.move_to_end(key)
        while len(self.d) > self.size:
            self.d.popitem(last=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    frac = float(os.environ.get("HM_SAMPLE_FRAC", "0.1"))
    world = workspace.setup(
        sample_frac=frac, seed=0, data_dir=os.environ.get("HM_DATA_DIR", "data")
    )
    app.state.system = System.build(world)
    app.state.cache = LRU(int(os.environ.get("HM_CACHE_SIZE", "4096")))
    app.state.index_of = {str(c): i for i, c in enumerate(world.hm.user_ids)}
    app.state.synthetic = world.synthetic
    app.state.images = ImageStore()
    yield


app = FastAPI(title="H&M two-stage recommender", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict:
    s: System = request.app.state.system
    return {
        "status": "ok",
        "synthetic_data": request.app.state.synthetic,
        "customers": s.world.hm.n_users,
        "articles": s.world.hm.n_items,
        "ranker_trees": s.ranker.best_iteration,
        "cache": {"hits": request.app.state.cache.hits, "misses": request.app.state.cache.misses},
    }


@app.get("/customers")
def customers(request: Request, limit: int = Query(5, ge=1, le=100)) -> dict:
    s: System = request.app.state.system
    users = s.world.ev_test.users[:limit]
    return {"customer_ids": [str(s.world.hm.user_ids[u]) for u in users]}


@app.get("/image/{article_id}")
def image(request: Request, article_id: int) -> Response:
    photo = request.app.state.images.get(article_id)
    if photo is None:
        raise HTTPException(404, "no photo for this article")
    return Response(photo, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@app.get("/recommend", response_model=RecommendResponse)
def recommend(
    request: Request,
    customer_id: str,
    k: int = Query(12, ge=1, le=50),
    lam: float = Query(1.0, ge=0.0, le=1.0, description="MMR: 1.0 = relevance only"),
) -> RecommendResponse:
    t0 = time.perf_counter()
    user = request.app.state.index_of.get(customer_id)
    if user is None:
        raise HTTPException(404, "unknown customer_id")
    key = (user, k, round(lam, 2))
    cache: LRU = request.app.state.cache
    recs, cached = cache.get(key), True
    if recs is None:
        cached = False
        df = request.app.state.system.recommend(user, k=k, lam=lam)
        recs = [
            Recommendation(
                article_id=int(r.article_id),
                name=r.name,
                product_type=r.product_type,
                colour=r.colour,
                score=float(r.score),
                why=r.why,
            )
            for r in df.itertuples()
        ]
        cache.put(key, recs)
    return RecommendResponse(
        customer_id=customer_id,
        k=k,
        lam=lam,
        cached=cached,
        latency_ms=(time.perf_counter() - t0) * 1000,
        recommendations=recs,
    )
