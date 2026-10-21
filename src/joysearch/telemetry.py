"""Request-local stage timings, enabled only for explicit debug requests."""

from contextvars import ContextVar
from functools import wraps
from time import perf_counter

_current = ContextVar("joysearch_trace", default=None)


class RequestTrace:
    def __init__(self, enabled):
        self.enabled = enabled
        self.spans = []
        self.stack = []

    def __enter__(self):
        self.started = perf_counter()
        self.token = _current.set(self if self.enabled else None)
        return self

    def __exit__(self, *exc):
        _current.reset(self.token)


def result_count(result):
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, list):
        return len(result)
    if isinstance(result, dict):
        return sum(len(v) for v in result.values() if isinstance(v, list))
    if hasattr(result, "shape"):
        return int(result.shape[0])
    return None


def timed(owner, name, stage):
    original = getattr(owner, name)
    if getattr(original, "_joysearch_timed", False):
        return

    @wraps(original)
    def call(*args, **kwargs):
        trace = _current.get()
        if trace is None:
            return original(*args, **kwargs)
        span = {
            "stage": stage,
            "span_id": len(trace.spans),
            "parent_id": trace.stack[-1] if trace.stack else None,
            "start_ms": (perf_counter() - trace.started) * 1000,
        }
        trace.spans.append(span)
        trace.stack.append(span["span_id"])
        started = perf_counter()
        try:
            result = original(*args, **kwargs)
            span["output_count"] = result_count(result)
            return result
        finally:
            span["elapsed_ms"] = (perf_counter() - started) * 1000
            trace.stack.pop()

    call._joysearch_timed = True
    setattr(owner, name, call)


def instrument(search, rec):
    import joysearch.recommendation.pipeline as rec_module
    import joysearch.search.pipeline as search_module

    if search:
        timed(search, "search", "search.total")
        timed(search.index, "retrieve", "search.bm25")
        timed(search.query_understanding, "parse", "search.parse")
        timed(search.catalog_attributes, "rerank", "search.attribute_sort")
        if search.dense:
            timed(search.dense, "retrieve", "search.dense")
            timed(search.dense.encoder, "encode", "search.encode")
            timed(search.dense, "rank_vector", "search.exact_scan_sort")
        if search.ranker:
            timed(search.features, "transform", "search.features")
            timed(search.ranker, "rerank", "search.ltr_sort")
        timed(search_module, "reciprocal_rank_fusion", "search.fusion")
    if rec:
        timed(rec, "recommend", "rec.total")
        timed(rec, "recall_channels", "rec.recall")
        timed(rec.popularity, "recommend", "rec.popularity")
        timed(rec.itemcf, "retrieve_native", "rec.itemcf_native")
        timed(rec.itemcf, "recommend", "rec.itemcf_served")
        if rec.two_tower:
            timed(rec.two_tower, "recommend", "rec.two_tower_exact")
        timed(rec_module, "filter_channels", "rec.filter")
        timed(rec_module, "allocate", "rec.allocation_sort")
