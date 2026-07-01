"""Integration tests for health and readiness endpoints."""
import sys
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

import app.models.model_store as ms
from app.main import app


@pytest.fixture(scope="module", autouse=True)
def load_models_once():
    """Load models once for the entire module — mirrors production startup."""
    ms.load_models()


# ── /health ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_returns_200():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"


@pytest.mark.asyncio
async def test_health_reports_models_loaded():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
    assert r.json()["models_loaded"] is True


@pytest.mark.asyncio
async def test_health_has_timestamp():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
    assert "timestamp" in r.json()


# ── /ready ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ready_returns_200_when_models_loaded():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["models_loaded"] is True


@pytest.mark.asyncio
async def test_ready_returns_503_when_not_ready(monkeypatch):
    """Simulate model-load failure by temporarily overriding is_ready."""
    monkeypatch.setattr(ms, "is_ready", lambda: False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "not_ready"
    assert body["models_loaded"] is False
