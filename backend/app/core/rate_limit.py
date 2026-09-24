"""In-process sliding-window rate limiting.

Scoped as a FastAPI dependency (not global middleware) so it only applies
to the routes that actually need it: letter prediction/event traffic and calibration requests.

This is deliberately an in-process limiter, not a Redis-backed distributed
one: this app runs as a single backend instance on Render (see render.yaml
and AGENTS.md's documented deployment shape), so there is no second
process for a shared limiter to coordinate with. If this backend is ever
scaled to multiple instances, this limiter's counters stop being accurate
across instances and it should be swapped for a shared store (Redis, etc.)
at that point — noted here so that assumption isn't silently forgotten.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status

from app.core.security import decode_access_token


class SlidingWindowRateLimiter:
    """Tracks request timestamps per client key over a rolling window.

    Thread-safe: FastAPI's sync route handlers run in a threadpool, so the
    shared state needs a lock, not just single-threaded-asyncio safety.
    """

    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._last_cleanup = 0.0
        self._cleanup_interval = min(window_seconds, 30.0)

    def check(self, key: str) -> tuple[bool, float]:
        """Returns (allowed, retry_after_seconds). Records the hit only if allowed."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] < cutoff:
                hits.popleft()

            if now - self._last_cleanup >= self._cleanup_interval:
                stale = [
                    client_key
                    for client_key, client_hits in self._hits.items()
                    if not client_hits or client_hits[-1] < cutoff
                ]
                for client_key in stale:
                    self._hits.pop(client_key, None)
                self._last_cleanup = now
                hits = self._hits[key]
            if len(hits) >= self.limit:
                retry_after = hits[0] + self.window_seconds - now
                return False, max(retry_after, 0.0)
            hits.append(now)
            return True, 0.0

    def reset(self) -> None:
        """Test-only: clear all tracked state between test cases."""
        with self._lock:
            self._hits.clear()


def _client_key(request: Request) -> str:
    """Authenticated requests are limited per user; invalid or missing bearer
    credentials fall back to client IP so fabricated tokens cannot bypass the
    per-user bucket by creating an unbounded number of token keys."""
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        subject = decode_access_token(auth_header[7:])
        if subject:
            return f"user:{subject}"
    client = request.client
    return f"ip:{client.host if client else 'unknown'}"


def make_rate_limit_dependency(limiter: SlidingWindowRateLimiter):
    """Build a FastAPI dependency bound to one limiter instance.

    The letter recognition and calibration routes use separate limiter
    instances so their counters and configured limits stay independent.
    """

    def _dependency(request: Request) -> None:
        allowed, retry_after = limiter.check(_client_key(request))
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Please slow down and try again shortly.",
                headers={"Retry-After": str(max(1, round(retry_after)))},
            )

    return _dependency
