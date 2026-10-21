from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from joysearch.api.schemas import RecRequest, SearchRequest, SearchResponse
from joysearch.common import ROOT, RecommendationContext, read_json
from joysearch.recommendation.pipeline import RecPipeline
from joysearch.search.pipeline import SearchPipeline


def create_app(root: Path = ROOT, enable_dense: bool = True, enable_two_tower: bool = True):
    @asynccontextmanager
    async def lifespan(app):
        search_path = root / "artifacts/search"
        app.state.search = (
            SearchPipeline.load(root, enable_dense=enable_dense) if search_path.exists() else None
        )
        rec_path = root / "artifacts/recommendation"
        app.state.rec = (
            RecPipeline.load(rec_path, enable_two_tower=enable_two_tower) if rec_path.exists() else None
        )
        yield

    app = FastAPI(title="JoySearch", lifespan=lifespan)

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    @app.get("/readyz")
    def ready():
        states = {
            "search_ready": app.state.search is not None,
            "rec_ready": app.state.rec is not None,
        }
        return JSONResponse(
            {
                **states,
                "search_algorithms": app.state.search.algorithms if app.state.search else [],
                "recommendation_algorithms": app.state.rec.algorithms if app.state.rec else [],
                "two_tower_ready": app.state.rec is not None and app.state.rec.two_tower is not None,
                "dense_ready": app.state.search is not None and app.state.search.dense is not None,
                "ltr_ready": app.state.search is not None and app.state.search.ranker is not None,
                "versions": {
                    "search": app.state.search.version if app.state.search else None,
                    "rec": app.state.rec.version if app.state.rec else None,
                },
                "reason": None if all(states.values()) else "Build missing model artifacts",
            },
            status_code=200 if all(states.values()) else 503,
        )

    @app.post(
        "/api/v1/search",
        response_model=SearchResponse,
        response_model_exclude_none=True,
        responses={503: {"description": "Requested search model is unavailable"}},
    )
    def search(request: SearchRequest):
        pipeline = app.state.search
        if pipeline is None:
            raise HTTPException(503, "Search index missing; run scripts/build_search.py")
        if request.algorithm not in pipeline.algorithms:
            script = "train_ranker.py" if request.algorithm == "ltr" else "build_dense.py"
            raise HTTPException(503, f"{request.algorithm} unavailable; run scripts/{script}")
        started = perf_counter()
        candidates = pipeline.search(request.query, request.top_k, request.algorithm)
        results = []
        for c in candidates:
            metadata = pipeline.metadata[c.item_id]
            results.append(
                {
                    "catalog_id": c.catalog_id,
                    "item_id": c.item_id,
                    "title": metadata["product_name"],
                    "score": c.score,
                    "rank": c.rank,
                    "sources": c.evidence.get("sources", [c.source]),
                    **({"evidence": c.evidence} if request.debug else {}),
                    "metadata": {"product_class": metadata["product_class"]},
                }
            )
        return {
            "catalog_id": request.catalog_id,
            "algorithm": request.algorithm,
            "model_version": pipeline.versions[request.algorithm],
            "results": results,
            "degraded": False,
            "trace_id": uuid4().hex,
            "latency_ms": (perf_counter() - started) * 1000,
        }

    @app.get("/api/v1/users/demo")
    def demo_users():
        if app.state.rec is None:
            raise HTTPException(503, "Recommendation model missing; run scripts/build_rec.py")
        return {
            "catalog_id": "amazon_baby_rec",
            "users": [
                {"user_id": alias, "history_count": len(app.state.rec.popularity.histories[user])}
                for alias, user in list(app.state.rec.aliases.items())[:20]
            ],
        }

    @app.post("/api/v1/recommend")
    def recommend(request: RecRequest):
        pipeline = app.state.rec
        if pipeline is None:
            raise HTTPException(503, "Recommendation model missing; run scripts/build_rec.py")
        if request.algorithm not in pipeline.algorithms:
            raise HTTPException(503, "two_tower unavailable; run scripts/train_two_tower.py")
        started = perf_counter()
        user = pipeline.aliases.get(request.user_id)
        candidates, reason = pipeline.recommend(
            RecommendationContext(user, exclude_seen=request.exclude_seen),
            request.top_k,
            request.algorithm,
        )
        return {
            "catalog_id": request.catalog_id,
            "algorithm": request.algorithm,
            "model_version": pipeline.versions[request.algorithm],
            "results": [
                {
                    "catalog_id": c.catalog_id,
                    "item_id": c.item_id,
                    "title": None,
                    "rank": c.rank,
                    "score": c.score,
                    "sources": [c.source],
                    "metadata": {"availability": "metadata unavailable"},
                }
                for c in candidates
            ],
            "degraded": reason is not None,
            "fallback_reason": reason,
            "trace_id": uuid4().hex,
            "latency_ms": (perf_counter() - started) * 1000,
        }

    @app.get("/api/v1/reports/summary")
    def reports():
        latest = root / "reports/latest.json"
        if not latest.exists():
            return {"status": "not_run", "reason": "Run scripts/evaluate.py first"}
        pointer = read_json(latest)
        folder = root / "reports" / pointer["run_id"]
        return {
            "status": "completed",
            "run_id": pointer["run_id"],
            "search": read_json(folder / "search_metrics.json"),
            "recommendation": read_json(folder / "recommendation_metrics.json"),
        }

    return app


app = create_app()
