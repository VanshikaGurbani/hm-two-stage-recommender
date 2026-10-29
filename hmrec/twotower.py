"""Two-tower retrieval model served through a FAISS HNSW index.

* item tower: article id embedding + category tags + text embedding -> MLP -> unit vector;
* user tower: mean of the embeddings of the customer's last ``hist_len`` purchases -> MLP -> unit
  vector, so any customer (even one never seen in training) can be embedded from purchases alone;
* loss: in-batch sampled softmax with a log-Q correction for popularity.

Trained on CPU. ``score`` is exact (dense matmul, used for evaluation); ``retrieve`` goes through
the ANN index (used for serving) and ``ann_recall`` measures what approximation costs.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ann import ItemIndex
from .models import FitContext, Recommender


def build_samples(frame, n_users: int, hist_len: int) -> tuple[np.ndarray, np.ndarray]:
    """(history, target) pairs: every purchase predicted from the customer's previous purchases.

    ``history`` is ``(N, hist_len)`` of article indices, newest last, padded with -1 on the left.
    Vectorised: sort by (customer, time) and shift by 1..hist_len inside each customer.
    """
    f = frame.sort_values(["user", "ts"], kind="stable")
    user = f["user"].to_numpy()
    item = f["item"].to_numpy()
    n = len(f)
    hist = np.full((n, hist_len), -1, dtype=np.int32)
    for j in range(1, hist_len + 1):
        same = np.zeros(n, dtype=bool)
        same[j:] = user[j:] == user[:-j]
        col = hist_len - j
        hist[j:, col] = np.where(same[j:], item[:-j], -1)
    keep = hist[:, -1] >= 0  # needs at least one previous purchase
    return hist[keep], item[keep].astype(np.int64)


def last_history(frame, n_users: int, hist_len: int) -> np.ndarray:
    """(n_users, hist_len) newest-last array of each customer's most recent purchases (-1 pad)."""
    f = frame.sort_values(["user", "ts"], kind="stable")
    out = np.full((n_users, hist_len), -1, dtype=np.int32)
    tail = f.groupby("user", sort=False).tail(hist_len)
    pos = tail.groupby("user", sort=False).cumcount().to_numpy()
    size = tail.groupby("user", sort=False)["item"].transform("size").to_numpy()
    out[tail["user"].to_numpy(), hist_len - size + pos] = tail["item"].to_numpy()
    return out


class _Net(nn.Module):
    def __init__(self, n_items, dim, tags, text, hidden):
        super().__init__()
        self.n_items = n_items
        self.hist_emb = nn.Embedding(n_items + 1, dim, padding_idx=n_items)
        self.id_emb = nn.Embedding(n_items, dim)
        self.register_buffer("tags", torch.as_tensor(tags))
        self.register_buffer("text", torch.as_tensor(text))
        self.tag_proj = nn.Linear(tags.shape[1], dim, bias=False)
        self.text_proj = nn.Linear(text.shape[1], dim, bias=False)
        self.item_mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.user_mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        for e in (self.hist_emb, self.id_emb):
            nn.init.normal_(e.weight, std=0.05)

    def items(self, idx):
        x = self.id_emb(idx) + self.tag_proj(self.tags[idx]) + self.text_proj(self.text[idx])
        return F.normalize(self.item_mlp(x), dim=-1)

    def users(self, hist):
        h = torch.where(hist < 0, torch.full_like(hist, self.n_items), hist)
        mask = (hist >= 0).float().unsqueeze(-1)
        pooled = (self.hist_emb(h) * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return F.normalize(self.user_mlp(pooled), dim=-1)


class TwoTower(Recommender):
    def __init__(
        self,
        dim: int = 64,
        hidden: int = 128,
        hist_len: int = 20,
        epochs: int = 6,
        batch: int = 4096,
        lr: float = 2e-3,
        temperature: float = 0.07,
        days: int = 180,
        max_samples_per_epoch: int = 1_000_000,
        seed: int = 0,
        verbose: bool = False,
    ):
        self.dim, self.hidden, self.hist_len = dim, hidden, hist_len
        self.epochs, self.batch, self.lr, self.tau = epochs, batch, lr, temperature
        self.days, self.cap, self.seed, self.verbose = days, max_samples_per_epoch, seed, verbose
        self.name = "two-tower"
        self.log: list[dict] = []

    def fit(self, ctx: FitContext):
        torch.manual_seed(self.seed)
        torch.set_num_threads(min(12, torch.get_num_threads()))
        data = ctx.data
        self.ctx, self.n_items = ctx, data.n_items
        frame = ctx.recent(self.days)
        hist, target = build_samples(frame, data.n_users, self.hist_len)
        self.net = _Net(data.n_items, self.dim, data.tag_matrix(), self._text(ctx), self.hidden)
        hist_t, target_t = torch.as_tensor(hist), torch.as_tensor(target)
        pop = torch.as_tensor(np.bincount(target, minlength=data.n_items), dtype=torch.float32)
        logq = torch.log((pop + 1.0) / (pop.sum() + data.n_items))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        g = torch.Generator().manual_seed(self.seed)
        n = len(target)
        for ep in range(1, self.epochs + 1):
            t0, total, steps = time.time(), 0.0, 0
            self.net.train()
            perm = torch.randperm(n, generator=g)[: self.cap]
            for s in range(0, len(perm) - self.batch + 1, self.batch):
                idx = perm[s : s + self.batch]
                tgt = target_t[idx]
                u = self.net.users(hist_t[idx].long())
                v = self.net.items(tgt)
                logits = (u @ v.T) / self.tau - logq[tgt][None, :]
                dup = (tgt[:, None] == tgt[None, :]) & ~torch.eye(len(tgt), dtype=torch.bool)
                loss = F.cross_entropy(logits.masked_fill(dup, -1e9), torch.arange(len(tgt)))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total, steps = total + loss.item(), steps + 1
            rec = {
                "epoch": ep,
                "loss": round(total / max(steps, 1), 4),
                "sec": round(time.time() - t0, 1),
            }
            self.log.append(rec)
            if self.verbose:
                print(rec, flush=True)
        self.net.eval()
        with torch.no_grad():
            self.item_vecs = self.net.items(torch.arange(data.n_items)).numpy()
        self.user_hist = last_history(ctx.frame, data.n_users, self.hist_len)
        self.index: ItemIndex | None = None
        return self

    @staticmethod
    def _text(ctx: FitContext) -> np.ndarray:
        emb = ctx.extras.get("text_embeddings") if hasattr(ctx, "extras") else None
        if emb is None:
            raise ValueError(
                "TwoTower needs ctx.extras['text_embeddings'] (item_embeddings output)"
            )
        return emb

    def user_vectors(self, hist: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.net.users(torch.as_tensor(hist, dtype=torch.long)).numpy()

    def score(self, users):
        return self.user_vectors(self.user_hist[users]) @ self.item_vecs.T

    # ------------------------------------------------------------------ ANN serving
    def build_index(self, kind: str = "hnsw", **kw) -> ItemIndex:
        self.index = ItemIndex(self.item_vecs, kind, **kw)
        return self.index

    def retrieve(self, users: np.ndarray, n: int) -> np.ndarray:
        """Top-n article indices per customer through the ANN index."""
        if self.index is None:
            self.build_index()
        assert self.index is not None
        _, ids = self.index.search(self.user_vectors(self.user_hist[users]), n)
        return ids

    def ann_source(self, n: int = 100) -> Recommender:
        """A candidate source whose scores come only from ANN retrieval (what serving uses)."""
        parent = self

        class _Ann(Recommender):
            name = "two-tower-ann"

            def score(self, users):
                q = parent.user_vectors(parent.user_hist[users])
                if parent.index is None:
                    parent.build_index()
                assert parent.index is not None
                sc, ids = parent.index.search(q, n)
                out = np.full((len(users), parent.n_items), -np.inf, dtype=np.float32)
                rows = np.repeat(np.arange(len(users)), ids.shape[1])
                ok = ids.ravel() >= 0
                out[rows[ok], ids.ravel()[ok]] = sc.ravel()[ok]
                return out

        return _Ann()

    def ann_recall(self, users: np.ndarray, n: int = 100) -> float:
        """Overlap of the HNSW top-n with the exact top-n, averaged over customers."""
        exact = ItemIndex(self.item_vecs, "flat")
        q = self.user_vectors(self.user_hist[users])
        _, e = exact.search(q, n)
        approx = self.retrieve(users, n)
        return float(np.mean([len(set(a) & set(b)) / n for a, b in zip(approx, e, strict=True)]))
