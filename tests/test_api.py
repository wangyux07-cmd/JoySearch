from fastapi.testclient import TestClient

from joysearch.api.main import create_app


def test_health_and_honest_readiness(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        response = client.get("/readyz")
        assert response.status_code == 503
        assert not response.json()["search_ready"]
        assert not response.json()["rec_ready"]


def test_real_search_api():
    with TestClient(create_app()) as client:
        response = client.post("/api/v1/search", json={"query": "salon chair", "top_k": 3})
        assert response.status_code == 200
        data = response.json()
        assert data["catalog_id"] == "wands_search"
        assert len(data["results"]) == 3
        assert data["results"][0]["title"]
        for invalid in [
            {"query": " "},
            {"query": "desk", "algorithm": "unknown"},
            {"query": "desk", "catalog_id": "amazon_baby_rec"},
            {"query": "desk", "top_k": 0},
        ]:
            assert client.post("/api/v1/search", json=invalid).status_code == 422


def test_recommendation_api_and_unready(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.post("/api/v1/search", json={"query": "chair"}).status_code == 503
        assert client.post("/api/v1/recommend", json={}).status_code == 503
    with TestClient(create_app()) as client:
        assert client.get("/readyz").status_code == 200
        users = client.get("/api/v1/users/demo").json()["users"]
        assert users and all(u["user_id"].startswith("anonymous_user_") for u in users)
        response = client.post("/api/v1/recommend", json={"user_id": users[0]["user_id"]})
        data = response.json()
        assert response.status_code == 200 and len(data["results"]) == 10
        assert data["catalog_id"] == "amazon_baby_rec"
        cold = client.post("/api/v1/recommend", json={"user_id": "unknown"}).json()
        assert cold["fallback_reason"] == "cold_user"
        assert all(c["sources"] == ["popularity"] for c in cold["results"])
        assert client.post("/api/v1/recommend", json={"algorithm": "unknown"}).status_code == 422
        assert (
            client.post("/api/v1/recommend", json={"catalog_id": "wands_search"}).status_code == 422
        )


def test_disabled_dense_preserves_bm25():
    with TestClient(create_app(enable_dense=False)) as client:
        ready = client.get("/readyz").json()
        assert ready["search_ready"] and not ready["dense_ready"]
        assert ready["search_algorithms"] == ["bm25"]
        for algorithm in ["dense", "hybrid", "ltr"]:
            response = client.post(
                "/api/v1/search", json={"query": "chair", "algorithm": algorithm}
            )
            assert response.status_code == 503
        assert client.post("/api/v1/search", json={"query": "chair"}).status_code == 200


def test_disabled_two_tower_preserves_recommendation_baselines():
    with TestClient(create_app(enable_dense=False, enable_two_tower=False)) as client:
        ready = client.get("/readyz").json()
        assert ready["rec_ready"] and not ready["two_tower_ready"]
        assert ready["recommendation_algorithms"] == ["popularity", "itemcf"]
        response = client.post("/api/v1/recommend", json={"algorithm": "two_tower"})
        assert response.status_code == 503 and "train_two_tower.py" in response.json()["detail"]
        assert client.post("/api/v1/recommend", json={"algorithm": "itemcf"}).status_code == 200
