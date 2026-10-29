"""Shared setup for the notebooks and Streamlit app: data, split, contexts, embeddings."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import data as D
from . import embed
from . import models as Mo
from . import splits as S
from .evaluate import Evaluator


@dataclass
class World:
    hm: D.HMData
    split: S.Split
    emb: np.ndarray
    ctx_train: Mo.FitContext  # history = train weeks (for the validation week)
    ctx_fit: Mo.FitContext  # history = train + validation (for the test week)
    ev_val: Evaluator
    ev_test: Evaluator
    synthetic: bool
    data_dir: Path


def setup(
    sample_frac: float = 0.1,
    seed: int = 0,
    encoder: str = "minilm",
    history_weeks: int | None = 26,
    data_dir: str | Path = "data",
    allow_fixture: bool = True,
) -> World:
    """Load the real data if present; else (``allow_fixture``) fall back to a tiny synthetic set.

    The fallback exists so the notebooks run end-to-end on a fresh clone. It is flagged by
    ``World.synthetic`` and every notebook prints a banner when it is used.
    """
    root = Path(data_dir)
    synthetic = False
    if not (root / "customers.csv").exists():
        if D.find_archive() is not None:
            D.download(root)
        elif allow_fixture:
            root = Path(tempfile.mkdtemp(prefix="hm_fixture_"))
            D.make_fixture(root, n_customers=600, n_articles=120, n_weeks=16, seed=seed)
            sample_frac, history_weeks, encoder, synthetic = 1.0, 8, "tfidf", True
            print("WARNING: real H&M data not found - using a tiny SYNTHETIC fixture.")
        else:
            raise FileNotFoundError(D.SETUP_HELP.format(root=root))
    hm = D.load(root, sample_frac=sample_frac, seed=seed)
    split = S.weekly(hm.interactions, history_weeks=history_weeks)
    S.check_no_leakage(split)
    emb = embed.item_embeddings(hm, encoder, cache_dir=root / "cache")
    hist = split.fit_frame
    return World(
        hm=hm,
        split=split,
        emb=emb,
        ctx_train=Mo.FitContext(hm, split.train, seed),
        ctx_fit=Mo.FitContext(hm, hist, seed),
        ev_val=Evaluator(hm, split.train, split.val, content_vectors=emb),
        ev_test=Evaluator(hm, hist, split.test, content_vectors=emb),
        synthetic=synthetic,
        data_dir=root,
    )
