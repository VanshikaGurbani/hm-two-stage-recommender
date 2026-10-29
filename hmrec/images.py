"""Product photos, read on demand.

H&M ships one JPEG per article as ``images/<first 3 digits>/<10-digit article id>.jpg`` (about
28 GB, ~105k files, not every article has one). Photos are looked up in this order:

1. an extracted ``images/`` folder (project root or ``data/``);
2. the competition zip itself (read one member at a time - nothing is extracted).

Photos are used for the demo UI and API only; the ranking models never see them.
"""

from __future__ import annotations

import threading
import zipfile
from pathlib import Path

from .data import DATA_DIR, find_archive


def image_member(article_id: int) -> str:
    """Photo path relative to the dataset root, e.g. ``images/010/0108775015.jpg``."""
    code = f"{int(article_id):010d}"
    return f"images/{code[:3]}/{code}.jpg"


class ImageStore:
    def __init__(self, folders: tuple[Path, ...] | None = None, archive: Path | None = None):
        roots = folders if folders is not None else (Path("."), DATA_DIR)
        self.root = next((r / "images" for r in roots if (r / "images").is_dir()), None)
        self._zip: zipfile.ZipFile | None = None
        self._names: set[str] = set()
        self._lock = threading.Lock()
        if self.root is None:
            archive = archive or find_archive()
            if archive is not None:
                self._zip = zipfile.ZipFile(archive)
                self._names = {n for n in self._zip.namelist() if n.startswith("images/")}

    @property
    def available(self) -> bool:
        return self.root is not None or self._zip is not None

    @property
    def source(self) -> str:
        if self.root is not None:
            return f"folder {self.root}"
        return "competition zip" if self._zip is not None else "none"

    def get(self, article_id: int) -> bytes | None:
        """JPEG bytes for the article, or ``None`` if it has no photo."""
        member = image_member(article_id)
        if self.root is not None:
            path = self.root.parent / member
            return path.read_bytes() if path.exists() else None
        if self._zip is not None and member in self._names:
            with self._lock:  # ZipFile is not thread-safe
                return self._zip.read(member)
        return None
