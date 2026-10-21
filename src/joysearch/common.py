import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Candidate:
    catalog_id: str
    item_id: str
    score: float
    source: str
    rank: int
    evidence: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RecommendationContext:
    user_id: str | None
    recent_item_ids: tuple[str, ...] = ()
    exclude_seen: bool = True
