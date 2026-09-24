"""Unit tests for the request rate limiter."""
import time

from fastapi import HTTPException

from app.core.rate_limit import SlidingWindowRateLimiter, _client_key, make_rate_limit_dependency


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


def test_client_key_uses_user_subject_for_valid_bearer(monkeypatch):
    monkeypatch.setattr("app.core.rate_limit.decode_access_token", lambda token: "42")

    class Client:
        host = "203.0.113.5"

    class Request:
        headers = {"authorization": "Bearer valid-token"}
        client = Client()

    assert _client_key(Request()) == "user:42"


def test_client_key_falls_back_to_ip_for_invalid_bearer(monkeypatch):
    monkeypatch.setattr("app.core.rate_limit.decode_access_token", lambda token: None)

    class Client:
        host = "203.0.113.5"

    class Request:
        headers = {"authorization": "Bearer forged-token"}
        client = Client()

    assert _client_key(Request()) == "ip:203.0.113.5"


def test_letter_routes_use_separate_calibration_and_recognition_limits():
    from app.api import letter as letter_api
    from app.core.config import get_settings

    settings = get_settings()
    assert letter_api._calibration_limiter.limit == settings.CALIBRATION_RATE_LIMIT_PER_MINUTE
    assert letter_api._recognition_limiter.limit == settings.TRANSLATE_RATE_LIMIT_PER_MINUTE
