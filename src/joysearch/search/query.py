"""Finite, catalog-grounded English query rules shared by offline and online search."""

import re
import unicodedata
from dataclasses import asdict, dataclass, replace

from joysearch.common import ROOT, read_json


def normalize(query):
    return " ".join(unicodedata.normalize("NFKC", query).lower().split())


def phrase_pattern(phrase):
    return re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)")


@dataclass
class QueryPlan:
    original_query: str
    normalized_query: str
    category_hints: list[str]
    attributes: dict[str, list[str]]
    negations: list[dict]
    rewrites: list[str]
    unsupported_constraints: list[str]
    rule_ids: list[str]
    slot_status: dict[str, str]

    def to_dict(self):
        return asdict(self)


class QueryUnderstanding:
    def __init__(self, config=None):
        self.config = config or read_json(ROOT / "configs/query.json")
        self.patterns = {
            slot: [
                (v, re.compile(r"(?<!\w)" + re.escape(v) + r"s?(?!\w)"))
                if slot == "category" and not v.endswith("s")
                else (v, phrase_pattern(v))
                for v in sorted(values, key=len, reverse=True)
            ]
            for slot, values in {
                "category": self.config["categories"],
                **{s: self.config[s] for s in ["color", "material", "size"]},
            }.items()
        }

    def mentions(self, text, slot):
        found, covered = [], set()
        for value, pattern in self.patterns[slot]:
            for match in pattern.finditer(text):
                positions = set(range(*match.span()))
                if not positions & covered:
                    found.append((value, match.start(), match.end()))
                    covered.update(positions)
        return sorted(found, key=lambda m: m[1])

    def parse(self, query):
        text = normalize(query)
        negated = list(
            re.finditer(
                r"\b(?:without|not|no|exclude|excluding)\b.*?(?=\b(?:with|but)\b|[,;!?]|\.(?!\d)|$)",
                text,
            )
        )
        attributes, negations, rules = {}, [], ["normalize:nfkc-case-space"]
        categories = []
        for slot in self.patterns:
            positive = []
            for value, start, end in self.mentions(text, slot):
                negative = any(m.start() <= start < m.end() for m in negated)
                canonical = {"grey": "gray", "wooden": "wood", "twinxl": "twin xl"}.get(
                    value, value
                )
                if negative:
                    negations.append({"slot": slot, "value": canonical, "span": [start, end]})
                elif canonical not in positive:
                    positive.append(canonical)
            if slot == "category":
                categories = positive
            else:
                attributes[slot] = positive
            if positive:
                rules.append(f"extract:{slot}")
        for match in re.finditer(
            r"(?<!\w)\d+(?:\.\d+)?(?:\s*[x×]\s*\d+(?:\.\d+)?){0,2}"
            r"\s*(?:inches|inch|in\b|feet|foot|ft\b|cm\b|mm\b|[\"′″'])(?!\w)",
            text,
        ):
            value = match.group().strip()
            if any(m.start() <= match.start() < m.end() for m in negated):
                negations.append({"slot": "size", "value": value, "span": list(match.span())})
            else:
                attributes["size"].append(value)
                rules.append("extract:dimension-with-unit")
        for match in re.finditer(r"(?<!\w)\d+(?:\.\d+)?\s*[x×]\s*\d+(?:\.\d+)?(?!\w)", text):
            if not any(match.group() in v for v in attributes["size"]) and not any(
                m.start() <= match.start() < m.end() for m in negated
            ):
                attributes["size"].append(match.group())
                rules.append("extract:dimension-unit-unknown")
        unsupported = []
        for name, pattern in {
            "price": r"[$£€]|\b(?:under|below|less than)\s+\d|\b(?:cheap|budget|price)\b",
            "stock": r"\b(?:in stock|out of stock|availability|available now)\b",
            "delivery": r"\b(?:delivery|shipping|deliver|tomorrow|next day)\b",
        }.items():
            if re.search(pattern, text):
                unsupported.append(name)
                rules.append(f"unsupported:{name}")
        if negated:
            rules.append("negation:preserved")
            for match in negated:
                negations.append(
                    {"slot": "scope", "value": match.group(), "span": list(match.span())}
                )
        rewrite = text
        for source, target in self.config["aliases"].items():
            rewrite, count = phrase_pattern(source).subn(target, rewrite)
            if count:
                rules.append(f"alias:{source}->{target}")
        # Quote-to-inch equivalence preserves all numeric values and other query terms.
        rewrite, count = re.subn(r'(\d)\s*(?:"|″|\'\')', r"\1 inch", rewrite)
        if count:
            rules.append("unit:quote-to-inch")
        rewrites = [rewrite] if rewrite != text else []
        status = {
            slot: "recognized" if values else "unrecognized" for slot, values in attributes.items()
        }
        status["category"] = "recognized" if categories else "unrecognized"
        return QueryPlan(
            query,
            text,
            categories,
            attributes,
            negations,
            rewrites[: self.config["max_rewrites"]],
            unsupported,
            sorted(set(rules)),
            status,
        )


class CatalogAttributes:
    """Use explicit feature keys only; variant aggregation prevents safe hard filtering."""

    def __init__(self, products, parser):
        self.parser = parser
        self.values = {}
        for row in products[["product_id", "product_features"]].fillna("").itertuples(index=False):
            fields = {}
            for entry in row.product_features.split("|"):
                key, sep, value = entry.partition(":")
                if sep:
                    fields.setdefault(key.strip(), []).append(value.strip())
            self.values[row.product_id] = {
                slot: " | ".join(v for key in keys for v in fields.get(key, []))
                for slot, keys in parser.config["field_keys"].items()
            }

    def compare(self, plan, item_id):
        states, scores = {}, []
        for slot, requested in plan.attributes.items():
            negative = [n["value"] for n in plan.negations if n["slot"] == slot]
            field = self.values[item_id][slot]
            known = {v for v, _, _ in self.parser.mentions(field, slot)}
            known = {
                {"grey": "gray", "wooden": "wood", "twinxl": "twin xl"}.get(v, v) for v in known
            }
            if slot == "material" and known & {"solid wood", "manufactured wood", "teak"}:
                known.add("wood")
            if not requested and not negative:
                state = "not_requested"
            elif not field:
                state = "field_missing"
            elif not known or (
                slot == "size" and any(v not in self.parser.config["size"] for v in requested)
            ):
                state = "unrecognized"
            elif set(negative) & known or (requested and not set(requested) & known):
                state = "contradiction"
                scores.append(-1)
            else:
                state = "match"
                scores.append(1)
            states[slot] = state
        return states, sum(scores) / max(len(scores), 1)

    def rerank(self, plan, candidates, weight):
        scored = []
        for c in candidates:
            states, score = self.compare(plan, c.item_id)
            scored.append(
                replace(
                    c,
                    score=c.score + weight * score,
                    evidence={
                        **c.evidence,
                        "attribute_states": states,
                        "attribute_match_score": score,
                        "hard_filtered": False,
                    },
                )
            )
        scored.sort(key=lambda c: (-c.score, c.item_id))
        return [replace(c, rank=i) for i, c in enumerate(scored, 1)]
