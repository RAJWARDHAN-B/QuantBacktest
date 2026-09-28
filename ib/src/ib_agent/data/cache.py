"""On-disk HTTP body cache keyed by URL."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CacheEntry:
    body: bytes
    fetched_at: float


class FileCache:
    def __init__(self, root: Path, clock: Callable[[], float] = time.time) -> None:
        self._root = root
        self._clock = clock

    def _paths(self, key: str) -> tuple[Path, Path]:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        folder = self._root / digest[:2]
        return folder / f"{digest}.body", folder / f"{digest}.json"

    def get(self, key: str, max_age_seconds: float | None) -> CacheEntry | None:
        """Return the cached entry, or None if missing or older than `max_age_seconds`.

        `max_age_seconds=None` means the entry never expires.
        """
        body_path, meta_path = self._paths(key)
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            body = body_path.read_bytes()
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        if meta.get("key") != key:
            return None
        fetched_at = float(meta["fetched_at"])
        if max_age_seconds is not None and self._clock() - fetched_at >= max_age_seconds:
            return None
        return CacheEntry(body=body, fetched_at=fetched_at)

    def put(self, key: str, body: bytes) -> CacheEntry:
        body_path, meta_path = self._paths(key)
        body_path.parent.mkdir(parents=True, exist_ok=True)
        fetched_at = self._clock()
        _atomic_write(body_path, body)
        # Meta is written last so a crash mid-write leaves a cache miss, not a torn entry.
        _atomic_write(meta_path, json.dumps({"key": key, "fetched_at": fetched_at}).encode())
        return CacheEntry(body=body, fetched_at=fetched_at)


def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
