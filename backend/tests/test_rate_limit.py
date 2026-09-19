"""Tests for backend/app/core/rate_limit.py.

Covers the limiter's pure logic (fast, deterministic, no model needed) and
end-to-end proof that /translate and /calibration actually enforce their
configured limits and return 429 with a usable Retry-After header — not
just that the limiter class works in isolation.
"""
import time

from fastapi.testclient import TestClient

from app.api import calibration as calibration_module
from app.api import translate as translate_module
from app.core.rate_limit import SlidingWindowRateLimiter, make_rate_limit_dependency
from app.main import app


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
    assert allowed_b  # a different client key must not share client-a's budget


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
    from fastapi import HTTPException

    limiter = SlidingWindowRateLimiter(limit=1, window_seconds=60)
    dependency = make_rate_limit_dependency(limiter)

    class _FakeClient:
        host = "203.0.113.5"

    class _FakeRequest:
        headers: dict = {}
        client = _FakeClient()

    dependency(_FakeRequest())  # first call: allowed, no exception
    try:
        dependency(_FakeRequest())
        assert False, "expected HTTPException on the second call"
    except HTTPException as exc:
        assert exc.status_code == 429
        assert "Retry-After" in exc.headers
        assert int(exc.headers["Retry-After"]) >= 1


def test_translate_endpoint_enforces_rate_limit_and_returns_retry_after():
    """End-to-end: exceed the real /translate rate limit and confirm the
    actual HTTP response is 429 with Retry-After, not just that the
    limiter class works in isolation."""
    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=60)
    # The route captured the original _rate_limit dependency at decoration
    # time (module import) — patching the module attribute afterward
    # wouldn't affect the already-built route. dependency_overrides is
    # FastAPI's supported mechanism for swapping a dependency at test time.
    app.dependency_overrides[translate_module._rate_limit] = make_rate_limit_dependency(limiter)

    try:
        with TestClient(app) as client:
            payload = {
                "user_id": None, "adapter_id": None,
                "pose_keypoints": [[0.0] * 132],
                "face_keypoints": [[0.0] * 1404],
                "left_hand_keypoints": [[0.0] * 63],
                "right_hand_keypoints": [[0.0] * 63],
            }
            # A single correctly-shaped frame is valid input to this endpoint
            # (no minimum frame count is enforced server-side — FRAME_WINDOW
            # is a client-side UI buffering choice, not a contract). It
            # reaches the model and returns 200 as long as the checkpoint is
            # available; the rate limiter dependency runs regardless.
            r1 = client.post("/api/v1/translate", json=payload)
            r2 = client.post("/api/v1/translate", json=payload)
            r3 = client.post("/api/v1/translate", json=payload)
            for r in (r1, r2):
                assert r.status_code in (200, 503), r.text  # 503 only if no checkpoint is present in this environment
            assert r3.status_code == 429, r3.text  # limit exceeded regardless of model availability
            assert "Retry-After" in r3.headers
    finally:
        app.dependency_overrides.pop(translate_module._rate_limit, None)


def test_calibration_endpoint_enforces_rate_limit():
    """Same proof for /calibration, which requires auth — confirms the
    limiter dependency runs even ahead of/alongside the auth dependency
    rather than only mattering for anonymous traffic."""
    limiter = SlidingWindowRateLimiter(limit=1, window_seconds=60)
    app.dependency_overrides[calibration_module._rate_limit] = make_rate_limit_dependency(limiter)

    try:
        with TestClient(app) as client:
            payload = {
                "user_id": 1, "calibration_seconds": 1,
                "pose_keypoints": [[0.0] * 132],
                "face_keypoints": [[0.0] * 1404],
                "left_hand_keypoints": [[0.0] * 63],
                "right_hand_keypoints": [[0.0] * 63],
                "target_text": "hi",
            }
            r1 = client.post("/api/v1/calibration", json=payload)  # no auth -> 401, but consumes the budget
            r2 = client.post("/api/v1/calibration", json=payload)
            assert r1.status_code == 401
            assert r2.status_code == 429, r2.text
            assert "Retry-After" in r2.headers
    finally:
        app.dependency_overrides.pop(calibration_module._rate_limit, None)
