import json

import pytest

from joysearch.common import ROOT, read_json
from joysearch.evaluation.llm_eval import classification_metrics, ranked_candidates
from joysearch.search.llm import (
    label_messages,
    product_document,
    rewrite_messages,
    validate_label,
    validate_rewrite,
)
from joysearch.search.query import QueryUnderstanding


def response(value):
    return {"raw": json.dumps(value), "stop_reason": "eos"}


def test_rewrite_constraints_cited_terms_and_strict_json():
    parser = QueryUnderstanding()
    query = "MOEN chair 18 inch 20 cm without red under £100"
    template = json.loads(rewrite_messages(query, [], parser)[1]["content"])["safe_template"]
    assert not validate_rewrite(query, response(template), [], parser)[1]
    for rewrite, expected in [
        (query.replace("MOEN", "Kohler"), "constraint_terms"),
        (query.replace("18 inch 20 cm", "18 cm 20 inch"), "constraint_units_or_currency"),
        (query.replace("without red", "with red"), "constraint_negation"),
        (query + " blue", "unsupported_rewrite_terms"),
    ]:
        _, errors = validate_rewrite(
            query, response({**template, "rewrites": [rewrite]}), [], parser
        )
        assert expected in errors
    docs = [
        {"product_id": "a", "product_class": "chair"},
        {"product_id": "b", "product_class": "seating"},
    ]
    proposal = {**template, "rewrites": [query + " seating"], "evidence_ids": ["a"]}
    assert (
        "unsupported_rewrite_terms" in validate_rewrite(query, response(proposal), docs, parser)[1]
    )
    proposal["evidence_ids"] = ["b"]
    assert not validate_rewrite(query, response(proposal), docs, parser)[1]
    assert (
        "invalid_json"
        in validate_rewrite(query, {"raw": "```{} ```", "stop_reason": "eos"}, [], parser)[1]
    )
    assert (
        "invalid_schema"
        in validate_rewrite(query, response({**template, "unknown": True}), [], parser)[1]
    )
    assert (
        "deadline_exceeded"
        in validate_rewrite(
            query, {**response(template), "stop_reason": "deadline_exceeded"}, [], parser
        )[1]
    )


def test_label_input_hides_gold_and_requires_literal_product_evidence():
    document = {
        "product_id": "abc",
        "product_name": "oak chair",
        "product_class": "chair",
        "product_description": "",
        "product_features": "",
    }
    messages = label_messages("chair", document)
    payload = json.loads(messages[1]["content"])
    assert set(payload) == {"query", "product", "allowed_evidence"}
    value = {"grade": 2, "uncertain": False, "evidence": payload["allowed_evidence"]}
    assert not validate_label(response(value), document)[1]
    value["evidence"][0]["quote"] = "solid mahogany"
    assert validate_label(response(value), document)[1] == ["unsupported_evidence_quote"]
    assert "invalid_schema" in validate_label(response({**value, "grade": True}), document)[1]
    assert not validate_label(
        response({"grade": None, "uncertain": True, "evidence": []}), document
    )[1]


def test_label_metrics_keep_abstentions_in_full_population():
    scores = classification_metrics([0, 1, 2, 2], [0, None, 1, 2])
    assert scores["confusion_gold_rows_predicted_0_1_2_abstain_columns"] == [
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 1, 0],
    ]
    assert scores["macro_f1_full_cohort_abstentions_as_fn"] == pytest.approx(5 / 9)
    assert scores["quadratic_weighted_kappa_accepted_only"] == pytest.approx(0.8)
    assert scores["abstention_count"] == scores["manual_review_queue_count"] == 1
    abstain = classification_metrics([0, 1, 2], [None] * 3)
    assert abstain["macro_f1_full_cohort_abstentions_as_fn"] == 0
    assert abstain["quadratic_weighted_kappa_accepted_only"] is None


def test_real_train_generation_cache_guard_and_fixed_budget():
    config = read_json(ROOT / "configs/llm.json")
    if not (ROOT / config["model_path"]).exists():
        pytest.skip("Optional local LLM has not been prepared")
    from joysearch.search.local_model import LocalGenerator
    from joysearch.search.pipeline import SearchPipeline

    model = SearchPipeline.load(ROOT, enable_ltr=False)
    query = "unique coffee tables"
    original, base, plan = model.query_candidates(query, "original")
    products = model.index.products.fillna("").set_index("product_id", drop=False)
    docs = [product_document(products.loc[c.item_id].to_dict(), config) for c in original[:3]]
    generator = LocalGenerator(ROOT)
    messages = rewrite_messages(query, docs, model.query_understanding)
    first = generator.generate(messages, config["rewrite_max_new_tokens"])
    second = generator.generate(messages, config["rewrite_max_new_tokens"])
    assert second["cache_hit"] and first["raw"] == second["raw"]
    proposal, errors = validate_rewrite(query, second, docs, model.query_understanding)
    rewrites = plan.rewrites if errors else proposal.rewrites
    pool = ranked_candidates(model, base, rewrites, query)
    assert len(pool) == len({c.item_id for c in pool}) == 100
    assert all(c.catalog_id == "wands_search" for c in pool)
    assert set(base) == {"bm25", "dense"}
    if errors:
        assert [c.item_id for c in pool] == [c.item_id for c in model.query_candidates(query)[0]]
