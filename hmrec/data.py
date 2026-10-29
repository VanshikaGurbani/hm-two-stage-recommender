"""H&M Personalized Fashion Recommendations: download, convert, sample, index.

Credentials come from a git-ignored ``.env`` file (see ``.env.example``); they are handed to the
official ``kaggle`` CLI as environment variables and are never read back or printed here.

Layout under ``data/``: the three Kaggle CSVs (``transactions_train.csv``, ``customers.csv``,
``articles.csv``) plus ``transactions_train.parquet``, the 3.5 GB CSV streamed once into Parquet.
Training runs on a seeded fraction of customers so it fits a CPU and 15 GB of RAM. Images (28 GB)
are not used.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .env import has_kaggle_credentials, kaggle_env

COMPETITION = "h-and-m-personalized-fashion-recommendations"
FILES = ("transactions_train.csv", "customers.csv", "articles.csv")
DATA_DIR = Path("data")
TAG_COLUMNS = (
    "index_group_name",
    "product_group_name",
    "colour_group_name",
    "garment_group_name",
    "graphical_appearance_name",
)
SETUP_HELP = (
    "H&M data not found in {root}. Join the Kaggle competition '" + COMPETITION + "' and accept "
    "its rules, then either save the competition zip in the project folder or put your Kaggle "
    "token in a `.env` file (copy `.env.example`), and run `python -m hmrec.data download`."
)


@dataclass
class HMData:
    """Dense-indexed purchases plus article / customer tables.

    ``interactions`` columns: user, item (dense indices), ts (purchase day, epoch seconds), price,
    channel. Each row is one purchase; repeat purchases of the same article are kept.
    """

    items: pd.DataFrame  # one row per item index (sorted by article_id)
    customers: pd.DataFrame  # one row per user index
    interactions: pd.DataFrame
    user_ids: np.ndarray  # customer_id hashes
    item_ids: np.ndarray  # article_id

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def tags(self) -> list[str]:
        return sorted({t for ts in self.items["tags"] for t in ts})

    def tag_matrix(self) -> np.ndarray:
        """(n_items, n_tags) float32 multi-hot of article categories."""
        index = {t: j for j, t in enumerate(self.tags)}
        m = np.zeros((self.n_items, len(index)), dtype=np.float32)
        for i, ts in enumerate(self.items["tags"]):
            for t in ts:
                m[i, index[t]] = 1.0
        return m


def find_archive(
    search: tuple[Path, ...] = (Path("."), DATA_DIR, Path.home() / "Downloads"),
) -> Path | None:
    """A complete Kaggle zip saved by hand (browser download), if one is present."""
    for folder in search:
        for p in sorted(Path(folder).glob("*.zip")):
            name = p.name.lower()
            if ("h-and-m" in name or "hm" in name) and zipfile.is_zipfile(p):
                return p
    return None


def extract_archive(archive: Path, data_dir: Path = DATA_DIR) -> Path:
    """Copy only the three CSVs out of the competition zip (skips the ~28 GB of images).

    The archive is treated as untrusted: members are matched by exact file name and written to a
    destination we choose, never to a path taken from the archive.
    """
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        by_name = {Path(i.filename).name: i for i in zf.infolist() if not i.is_dir()}
        for name in FILES:
            if (root / name).exists():
                continue
            if name not in by_name:
                raise FileNotFoundError(f"{name} not found inside {archive.name}")
            with zf.open(by_name[name]) as src, open(root / name, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
    return root


def download(data_dir: Path = DATA_DIR, env_path: Path | str = ".env") -> Path:
    """Fetch the three CSVs with the Kaggle CLI using credentials from ``.env``."""
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    archive = find_archive()
    if archive is not None and not all((root / f).exists() for f in FILES):
        return extract_archive(archive, root)
    cli = shutil.which("kaggle")
    if cli is None:
        raise RuntimeError("kaggle CLI not found; run `pip install kaggle`.")
    missing = [f for f in FILES if not (root / f).exists()]
    if missing and not has_kaggle_credentials(env_path):
        raise RuntimeError(
            "No Kaggle credentials: create a `.env` file from `.env.example` and set "
            "KAGGLE_API_TOKEN (or KAGGLE_USERNAME + KAGGLE_KEY)."
        )
    for name in missing:
        zipped = root / f"{name}.zip"
        if not zipped.exists():
            subprocess.run(
                [cli, "competitions", "download", "-c", COMPETITION, "-f", name, "-p", str(root)],
                check=True,
                env=kaggle_env(env_path),
            )
        if zipped.exists():
            with zipfile.ZipFile(zipped) as zf:
                zf.extractall(root)
    return root


def convert_transactions(root: Path) -> Path:
    """Stream ``transactions_train.csv`` into Parquet once (idempotent)."""
    import pyarrow as pa
    import pyarrow.csv as pacsv
    import pyarrow.parquet as pq

    out = root / "transactions_train.parquet"
    if out.exists():
        return out
    src = root / "transactions_train.csv"
    if not src.exists():
        raise FileNotFoundError(SETUP_HELP.format(root=root))
    types = {
        "t_dat": pa.date32(),
        "customer_id": pa.string(),
        "article_id": pa.int64(),
        "price": pa.float32(),
        "sales_channel_id": pa.int8(),
    }
    reader = pacsv.open_csv(src, convert_options=pacsv.ConvertOptions(column_types=types))
    tmp = out.with_suffix(".tmp")
    with pq.ParquetWriter(tmp, reader.schema, compression="zstd") as writer:
        for batch in reader:
            writer.write_batch(batch)
    tmp.rename(out)
    return out


def sample_customers(customer_ids: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Seeded customer subsample; deterministic for a given (ids, frac, seed)."""
    ids = np.sort(np.unique(customer_ids))
    if frac >= 1.0:
        return ids
    rng = np.random.default_rng(seed)
    n = max(1, round(len(ids) * frac))
    return np.sort(rng.choice(ids, size=n, replace=False))


def _articles(articles: pd.DataFrame) -> pd.DataFrame:
    articles = articles.sort_values("article_id").reset_index(drop=True)
    tags = [
        [f"{c.replace('_name', '')}={row[c]}" for c in TAG_COLUMNS if pd.notna(row[c])]
        for _, row in articles[list(TAG_COLUMNS)].iterrows()
    ]
    text = (
        articles["prod_name"].fillna("")
        + ". "
        + articles["product_type_name"].fillna("")
        + ", "
        + articles["colour_group_name"].fillna("")
        + ", "
        + articles["department_name"].fillna("")
        + ". "
        + articles["detail_desc"].fillna("")
    ).str.strip()
    return pd.DataFrame(
        {
            "article_id": articles["article_id"].to_numpy(),
            "name": articles["prod_name"].fillna("").to_numpy(),
            "product_type": articles["product_type_name"].fillna("").to_numpy(),
            "product_group": articles["product_group_name"].fillna("").to_numpy(),
            "colour": articles["colour_group_name"].fillna("").to_numpy(),
            "department": articles["department_name"].fillna("").to_numpy(),
            "tags": tags,
            "text": text.to_numpy(),
        }
    )


def _customer_features(customers: pd.DataFrame) -> pd.DataFrame:
    age = customers["age"].astype(float)
    return pd.DataFrame(
        {
            "age": age.fillna(age.median()).to_numpy(),
            "age_missing": age.isna().astype(float).to_numpy(),
            "fn": customers["FN"].fillna(0.0).astype(float).to_numpy(),
            "active": customers["Active"].fillna(0.0).astype(float).to_numpy(),
            "club_active": (customers["club_member_status"] == "ACTIVE").astype(float).to_numpy(),
            "news_regularly": (customers["fashion_news_frequency"] == "Regularly")
            .astype(float)
            .to_numpy(),
        }
    )


def load(data_dir: Path = DATA_DIR, sample_frac: float = 0.1, seed: int = 0) -> HMData:
    """Load a seeded customer sample as dense-indexed :class:`HMData`."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    root = Path(data_dir)
    for name in ("customers.csv", "articles.csv"):
        if not (root / name).exists():
            raise FileNotFoundError(SETUP_HELP.format(root=root))
    parquet = convert_transactions(root)
    all_customers = pq.read_table(parquet, columns=["customer_id"])["customer_id"]
    keep = sample_customers(
        pc.unique(all_customers).to_numpy(zero_copy_only=False), sample_frac, seed
    )
    table = pq.read_table(parquet, filters=[("customer_id", "in", pa.array(keep.tolist()))])
    table = table.to_pandas()

    items = _articles(pd.read_csv(root / "articles.csv"))
    item_ids = items["article_id"].to_numpy()
    table = table[table["article_id"].isin(set(item_ids))]
    customers = pd.read_csv(root / "customers.csv").set_index("customer_id").loc[keep].reset_index()
    u_map = pd.Series(np.arange(len(keep)), index=keep)
    i_map = pd.Series(np.arange(len(item_ids)), index=item_ids)
    seconds = pd.to_datetime(table["t_dat"]).astype("datetime64[s]").astype("int64")
    inter = pd.DataFrame(
        {
            "user": u_map.loc[table["customer_id"]].to_numpy(),
            "item": i_map.loc[table["article_id"]].to_numpy(),
            "ts": seconds.to_numpy(),
            "price": table["price"].to_numpy(dtype=np.float32),
            "channel": table["sales_channel_id"].to_numpy(dtype=np.int8),
        }
    )
    inter = inter.sort_values(["ts", "item", "user"], kind="stable").reset_index(drop=True)
    return HMData(items, customers, inter, keep, item_ids)


def customer_features(data: HMData) -> pd.DataFrame:
    return _customer_features(data.customers)


def describe(d: HMData) -> dict:
    n = len(d.interactions)
    days = (d.interactions["ts"].max() - d.interactions["ts"].min()) / 86400
    return {
        "customers": d.n_users,
        "articles_in_catalogue": d.n_items,
        "articles_purchased": int(d.interactions["item"].nunique()),
        "purchases": n,
        "span_days": round(float(days), 1),
        "purchases_per_customer": round(n / d.n_users, 2),
    }


# ---------------------------------------------------------------------- synthetic fixture


def make_fixture(
    root: Path, n_customers: int = 240, n_articles: int = 90, n_weeks: int = 10, seed: int = 0
) -> Path:
    """Write a tiny dataset in the Kaggle schema with planted taste and repurchase structure.

    Customers belong to one of 3 style groups and mostly buy that group's articles; ~25% of
    purchases repeat an earlier purchase. Used by tests only; contains no real data.
    """
    rng = np.random.default_rng(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    groups = ["Ladieswear", "Menswear", "Sport"]
    colours = ["Black", "White", "Blue", "Red"]
    art = pd.DataFrame(
        {
            "article_id": 100000000 + np.arange(n_articles) * 7 + 1,
            "prod_name": [f"Item {i}" for i in range(n_articles)],
            "product_type_name": [f"Type{i % 5}" for i in range(n_articles)],
            "product_group_name": [groups[i % 3] for i in range(n_articles)],
            "graphical_appearance_name": ["Solid"] * n_articles,
            "colour_group_name": [colours[i % 4] for i in range(n_articles)],
            "department_name": [f"Dept{i % 6}" for i in range(n_articles)],
            "index_group_name": [groups[i % 3] for i in range(n_articles)],
            "garment_group_name": [f"Garment{i % 4}" for i in range(n_articles)],
            "detail_desc": [
                f"A {colours[i % 4].lower()} piece for {groups[i % 3]}" for i in range(n_articles)
            ],
        }
    )
    art.to_csv(root / "articles.csv", index=False)
    cust_ids = [f"{c:064x}" for c in rng.choice(2**62, n_customers, replace=False)]
    pd.DataFrame(
        {
            "customer_id": cust_ids,
            "FN": rng.choice([1.0, np.nan], n_customers),
            "Active": rng.choice([1.0, np.nan], n_customers),
            "club_member_status": rng.choice(["ACTIVE", "PRE-CREATE"], n_customers),
            "fashion_news_frequency": rng.choice(["Regularly", "NONE"], n_customers),
            "age": rng.integers(16, 70, n_customers).astype(float),
            "postal_code": ["x"] * n_customers,
        }
    ).to_csv(root / "customers.csv", index=False)
    start = pd.Timestamp("2020-06-01")
    rows = []
    for c in range(n_customers):
        mine = np.flatnonzero(np.arange(n_articles) % 3 == c % 3)
        bought: list[int] = []
        days = np.sort(rng.integers(0, n_weeks * 7, int(rng.integers(8, 25))))
        for d in days:
            if bought and rng.random() < 0.25:
                a = int(rng.choice(bought))
            elif rng.random() < 0.85:
                a = int(rng.choice(mine))
            else:
                a = int(rng.integers(0, n_articles))
            bought.append(a)
            rows.append(
                (
                    (start + pd.Timedelta(days=int(d))).date().isoformat(),
                    cust_ids[c],
                    int(art["article_id"].iloc[a]),
                    round(float(rng.uniform(0.01, 0.1)), 4),
                    int(rng.integers(1, 3)),
                )
            )
    pd.DataFrame(
        rows, columns=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
    ).to_csv(root / "transactions_train.csv", index=False)
    return root


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hmrec.data")
    ap.add_argument("action", choices=["download", "convert", "describe"])  # download also unzips
    ap.add_argument("--sample-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.action == "download":
        download()
        print("downloaded:", ", ".join(FILES))
    elif args.action == "convert":
        print(convert_transactions(DATA_DIR))
    else:
        for k, v in describe(load(sample_frac=args.sample_frac, seed=args.seed)).items():
            print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
