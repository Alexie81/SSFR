from __future__ import annotations


def test_fastapi_schema_contains_required_endpoints() -> None:
    from demos.api_demo import app

    paths = app.openapi()["paths"]
    assert "/health" in paths
    assert "/index/build" in paths
    assert "/search" in paths
    assert "/router/stats" in paths
