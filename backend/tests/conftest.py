import os
from pathlib import Path

TEST_DB = Path("/tmp/visionbridge-release-tests.db")
TEST_DB.unlink(missing_ok=True)

os.environ["ENV"] = "test"
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB}"
os.environ["SECRET_KEY"] = "test-secret"
os.environ["AUTH_COOKIE_SECURE"] = "false"
os.environ["AUTH_COOKIE_SAMESITE"] = "lax"
os.environ["ALLOWED_ORIGINS"] = "http://testserver"
