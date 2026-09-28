from __future__ import annotations

from pathlib import Path

from ib_agent.data.cache import FileCache


class FakeClock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_round_trip(tmp_path: Path) -> None:
    cache = FileCache(tmp_path)
    cache.put("https://example/a", b"hello")
    entry = cache.get("https://example/a", max_age_seconds=60)
    assert entry is not None and entry.body == b"hello"


def test_miss_for_unknown_key(tmp_path: Path) -> None:
    assert FileCache(tmp_path).get("nope", max_age_seconds=None) is None


def test_ttl_expiry(tmp_path: Path) -> None:
    clock = FakeClock()
    cache = FileCache(tmp_path, clock=clock)
    cache.put("k", b"v")
    clock.now += 59
    assert cache.get("k", max_age_seconds=60) is not None
    clock.now += 1
    assert cache.get("k", max_age_seconds=60) is None


def test_none_ttl_never_expires(tmp_path: Path) -> None:
    clock = FakeClock()
    cache = FileCache(tmp_path, clock=clock)
    cache.put("k", b"v")
    clock.now += 10**9
    assert cache.get("k", max_age_seconds=None) is not None


def test_zero_ttl_forces_refresh(tmp_path: Path) -> None:
    cache = FileCache(tmp_path, clock=FakeClock())
    cache.put("k", b"v")
    assert cache.get("k", max_age_seconds=0) is None


def test_missing_meta_is_a_miss(tmp_path: Path) -> None:
    cache = FileCache(tmp_path)
    cache.put("k", b"v")
    for meta in tmp_path.rglob("*.json"):
        meta.unlink()
    assert cache.get("k", max_age_seconds=None) is None


def test_overwrite(tmp_path: Path) -> None:
    cache = FileCache(tmp_path)
    cache.put("k", b"old")
    cache.put("k", b"new")
    entry = cache.get("k", max_age_seconds=None)
    assert entry is not None and entry.body == b"new"
