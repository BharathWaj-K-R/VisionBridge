"""Unit tests for the request rate limiter."""
import time

from fastapi import HTTPException

from app.core.rate_limit import SlidingWindowRateLimiter, make_rate_limit_dependency


def test_limiter_allows_up_to_the_limit_then_blocks():
    limiter = SlidingWindowRateLimiter(limit=3, window_seconds=60)
    for _ in range(3):
        allowed, _ = limiter.check("client-a")
        assert allowed
    allowed, retry_after = limiter.check("client-a")
    assert not allowed
    assert retry_after > 0


def test_limiter_tracks_clients_independently():
    limiter = SlidingWindowRateLimiter(limit=1, window_seconds=60)
    allowed_a, _ = limiter.check("client-a")
    allowed_b, _ = limiter.check("client-b")
    assert allowed_a
    assert allowed_b


def test_limiter_window_expires_old_hits():
    limiter = SlidingWindowRateLimiter(limit=1, window_seconds=0.05)
    allowed_first, _ = limiter.check("client-a")
    assert allowed_first
    allowed_immediately, _ = limiter.check("client-a")
    assert not allowed_immediately
    time.sleep(0.06)
    allowed_after_window, _ = limiter.check("client-a")
    assert allowed_after_window


def test_dependency_raises_429_with_retry_after_header():
    limiter = SlidingWindowRateLimiter(limit=1, window_seconds=60)
    dependency = make_rate_limit_dependency(limiter)

    class Client:
        host = "203.0.113.5"

    class Request:
        headers = {}
        client = Client()

    dependency(Request())

    try:
        dependency(Request())
    except HTTPException as exc:
        assert exc.status_code == 429
        assert int(exc.headers["Retry-After"]) >= 1
    else:
        raise AssertionError("expected HTTPException on the second call")
