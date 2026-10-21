import argparse
from contextlib import contextmanager

import httpx
from fastapi.testclient import TestClient

from joysearch.api.main import create_app


@contextmanager
def client_for(base_url):
    if base_url:
        with httpx.Client(base_url=base_url, timeout=30) as client:
            yield client
    else:
        with TestClient(create_app()) as client:
            yield client


def smoke(base_url=None):
    with client_for(base_url) as client:
        assert client.get("/healthz").status_code == 200
        ready = client.get("/readyz")
        assert ready.status_code == 200, ready.text
        search = client.post("/api/v1/search", json={"query": "salon chair", "top_k": 10})
        assert search.status_code == 200, search.text
        assert search.json()["catalog_id"] == "wands_search"
        assert len(search.json()["results"]) == 10
        if ready.json()["dense_ready"]:
            for algorithm in ["dense", "hybrid"]:
                semantic = client.post(
                    "/api/v1/search",
                    json={"query": "salon chair", "algorithm": algorithm, "debug": True},
                )
                assert semantic.status_code == 200, semantic.text
                assert len(semantic.json()["results"]) == 10
                assert semantic.json()["model_version"] in ["dense-v1", "rrf-v1"]
        if ready.json()["ltr_ready"]:
            ranked = client.post(
                "/api/v1/search", json={"query": "salon chair", "algorithm": "ltr", "debug": True}
            )
            assert ranked.status_code == 200, ranked.text
            assert ranked.json()["model_version"] == "lambdamart-v1"
            assert len(ranked.json()["results"]) == 10
            assert all("feature_values" in r["evidence"] for r in ranked.json()["results"])
        users = client.get("/api/v1/users/demo").json()["users"]
        rankings = []
        for user in users[:2]:
            rec = client.post(
                "/api/v1/recommend", json={"user_id": user["user_id"], "algorithm": "itemcf"}
            )
            assert rec.status_code == 200, rec.text
            assert rec.json()["catalog_id"] == "amazon_baby_rec"
            rankings.append([item["item_id"] for item in rec.json()["results"]])
        assert rankings[0] != rankings[1]
        cold = client.post("/api/v1/recommend", json={"user_id": "unknown"}).json()
        assert cold["fallback_reason"] == "cold_user"
        assert (
            client.post(
                "/api/v1/search", json={"query": "desk", "algorithm": "unknown"}
            ).status_code
            == 422
        )
        report = client.get("/api/v1/reports/summary").json()
        assert report["status"] == "completed"
        assert isinstance(report["search"]["metrics"]["ndcg_at_10"], float)
        assert report["recommendation"]["users_evaluated"] > 0
        print(
            {
                "status": "PASS",
                "transport": base_url or "in-process ASGI",
                "search_top_id": search.json()["results"][0]["item_id"],
                "anonymous_users_differ": True,
                "cold_fallback": "cold_user",
                "report_run_id": report["run_id"],
                "readiness": ready.json(),
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url")
    smoke(parser.parse_args().base_url)
