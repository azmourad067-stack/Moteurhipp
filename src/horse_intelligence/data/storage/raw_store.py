"""Stockage des réponses brutes, immuables, compressées et adressées par leur hash."""

from __future__ import annotations

import gzip
from pathlib import Path

from horse_intelligence.data.collectors.base import RawResponse


class RawStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, raw: RawResponse) -> Path:
        d = raw.fetched_at
        return (self.root / raw.source / f"{d:%Y}" / f"{d:%m}" / f"{d:%d}"
                / f"{raw.endpoint}-{raw.sha256}.json.gz")

    def write(self, raw: RawResponse) -> Path:
        path = self.path_for(raw)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with gzip.open(tmp, "wb") as fh:
                fh.write(raw.body)
            tmp.replace(path)
        return path

    @staticmethod
    def read(path: Path) -> bytes:
        with gzip.open(path, "rb") as fh:
            return fh.read()
