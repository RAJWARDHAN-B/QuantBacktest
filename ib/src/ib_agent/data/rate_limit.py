"""Minimum-interval rate limiter (thread-safe)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class RateLimiter:
    def __init__(
        self,
        requests_per_second: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self._interval = 1.0 / requests_per_second
        self._clock = clock
        self._sleep = sleep
        self._next_slot = float("-inf")
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = self._clock()
            start = max(now, self._next_slot)
            if start > now:
                self._sleep(start - now)
            self._next_slot = start + self._interval
