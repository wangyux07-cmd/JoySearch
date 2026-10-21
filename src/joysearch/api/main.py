import os
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from joysearch.api.schemas import RecRequest, SearchRequest, SearchResponse
from joysearch.common import ROOT, RecommendationContext, read_json
from joysearch.recommendation.pipeline import RecPipeline
from joysearch.search.pipeline import SearchPipeline
from joysearch.telemetry import RequestTrace, instrument


def create_app(
    root: Path = ROOT,
    enable_dense: bool = True,
    enable_two_tower: bool = True,
    worker_threads: int | None = None,
    inference_threads: int | None = None,
    scan_mode: str = "reference",
    enable_ltr: bool = True,
):
    if scan_mode not in {"reference", "cached_controller"}:
        raise ValueError(f"Unknown exact scan mode: {scan_mode}")

    @asynccontextmanager
    async def lifespan(app):
        loading = perf_counter()
        search_path = root / "artifacts/search"
        app.state.search = (
            SearchPipeline.load(
                root, enable_dense=enable_dense, enable_ltr=enable_ltr and scan_mode == "reference"
            )
            if search_path.exists()
            else None
        )
        rec_path = root / "artifacts/recommendation"
        app.state.rec = (
            RecPipeline.load(rec_path, enable_two_tower=enable_two_tower)
            if rec_path.exists()
            else None
        )
        if scan_mode == "cached_controller" and app.state.search and app.state.search.dense:
            from joysearch.search.exact_scan import CachedControllerDenseIndex

            app.state.search.dense = CachedControllerDenseIndex(app.state.search.dense)
        instrument(app.state.search, app.state.rec)
        if worker_threads is not None:
            import anyio.to_thread

            anyio.to_thread.current_default_thread_limiter().total_tokens = worker_threads
        if inference_threads is not None:
            import torch
            from threadpoolctl import threadpool_limits

            torch.set_num_threads(inference_threads)
            limits = threadpool_limits(limits=inference_threads)
        else:
            limits = nullcontext()
        app.state.runtime = {
            "pid": os.getpid(),
            "load_seconds": perf_counter() - loading,
            "worker_threads": worker_threads,
            "inference_threads": inference_threads,
            "response_cache": "off",
            "scan_mode": scan_mode,
            "stage_timings": "debug requests only",
        }
        with limits:
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
                "runtime": app.state.runtime,
                "search_algorithms": app.state.search.algorithms if app.state.search else [],
                "recommendation_algorithms": app.state.rec.algorithms if app.state.rec else [],
                "two_tower_ready": app.state.rec is not None
                and app.state.rec.two_tower is not None,
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
        with RequestTrace(request.debug) as trace:
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
            "query_plan": pipeline.query_understanding.parse(request.query).to_dict()
            if request.debug
            else None,
            "degraded": False,
            "timings": trace.spans if request.debug else None,
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
            script = (
                "evaluate_allocation.py --split validation --cache <E1-validation-run>"
                if request.algorithm in {"quota", "adaptive_quota"}
                else "train_two_tower.py"
            )
            raise HTTPException(503, f"{request.algorithm} unavailable; run scripts/{script}")
        started = perf_counter()
        user = pipeline.aliases.get(request.user_id)
        with RequestTrace(request.debug) as trace:
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
                    "sources": c.evidence.get("sources", [c.source]),
                    **({"evidence": c.evidence} if request.debug else {}),
                    "metadata": {"availability": "metadata unavailable"},
                }
                for c in candidates
            ],
            "degraded": reason is not None,
            **({"timings": trace.spans} if request.debug else {}),
            "fallback_reason": reason,
            "allocation": candidates[0].evidence.get("allocation") if candidates else None,
            "trace_id": uuid4().hex,
            "latency_ms": (perf_counter() - started) * 1000,
        }

    @app.get("/api/v1/reports/allocation")
    def allocation_report():
        latest = root / "reports/allocation_test_latest.json"
        if not latest.exists():
            return {"status": "not_run", "reason": "Run scripts/evaluate_allocation.py first"}
        folder = root / "reports" / read_json(latest)["run_id"]
        return {
            "status": "completed",
            "report": read_json(folder / "metrics.json"),
            "slices": read_json(folder / "slices.json"),
            "simulation": read_json(folder / "simulation_summary.json"),
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
