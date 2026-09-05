"""Smoke tests for the FastAPI service. Board-dependent checks skip when no board is built
(e.g. in CI), so the health/schema checks still run everywhere."""
import os

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from eldraft.api import app  # noqa: E402

client = TestClient(app)
HAS_BOARD = os.path.exists(os.path.join("data", "board.json"))


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_openapi_schema():
    assert client.get("/openapi.json").status_code == 200


@pytest.mark.skipif(not HAS_BOARD, reason="board not built")
def test_optimize_within_budget():
    r = client.post("/optimize", json={"budget": 100})
    assert r.status_code == 200
    body = r.json()
    assert body["cost"] <= 100.001
    assert len(body["squad"]) >= 1


@pytest.mark.skipif(not HAS_BOARD, reason="board not built")
def test_players_listing():
    r = client.get("/players?pos=C&limit=5")
    assert r.status_code == 200
    assert all(p["position"] == "C" for p in r.json()["players"])
