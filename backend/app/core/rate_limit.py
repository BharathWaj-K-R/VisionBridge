"""In-process sliding-window rate limiting.

Scoped as a FastAPI dependency (not global middleware) so it only applies
to the routes that actually need it — /translate and /calibration, the two
endpoints that run real model inference/training per request.

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

    def check(self, key: str) -> tuple[bool, float]:
        """Returns (allowed, retry_after_seconds). Records the hit only if allowed."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] < cutoff:
                hits.popleft()
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
    """Authenticated requests are limited per-user (so one signer can't be
    starved by another client sharing a NAT/proxy IP); anonymous requests
    fall back to client IP, which is the best identity available for the
    public demo path."""
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        # Keyed on the raw token, not the decoded user id — avoids a second
        # JWT decode here purely for rate-limit bucketing; a forged/garbage
        # token still gets its own (harmless, since it'll fail real auth
        # downstream) bucket rather than falling through to a shared one.
        return f"token:{auth_header[7:]}"
    client = request.client
    return f"ip:{client.host if client else 'unknown'}"


def make_rate_limit_dependency(limiter: SlidingWindowRateLimiter):
    """Builds a FastAPI dependency bound to a specific limiter instance —
    lets /translate and /calibration each have their own limit/window
    without sharing counters."""

    def _dependency(request: Request) -> None:
        allowed, retry_after = limiter.check(_client_key(request))
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Please slow down and try again shortly.",
                headers={"Retry-After": str(max(1, round(retry_after)))},
            )

    return _dependency
