from __future__ import annotations

import pytest

from ib_agent.data.rate_limit import RateLimiter


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_spaces_requests_at_configured_rate() -> None:
    t = FakeTime()
    limiter = RateLimiter(5, clock=t.clock, sleep=t.sleep)
    for _ in range(3):
        limiter.acquire()
    assert t.sleeps == pytest.approx([0.2, 0.2])


def test_no_sleep_when_requests_are_already_spaced() -> None:
    t = FakeTime()
    limiter = RateLimiter(5, clock=t.clock, sleep=t.sleep)
    limiter.acquire()
    t.now += 1.0
    limiter.acquire()
    assert t.sleeps == []


def test_rejects_non_positive_rate() -> None:
    with pytest.raises(ValueError):
        RateLimiter(0)
