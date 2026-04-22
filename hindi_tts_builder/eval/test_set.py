"""Locked evaluation test set.

A TestSet is a curated collection of Hindi sentences organized into categories.
Once saved, it is treated as immutable — you should not change it between runs
or your baseline comparisons become invalid.

Default categories (matching what we discussed throughout this project):

    narration       100 sentences   — long-form storyteller prose
    dialogue        100 sentences   — questions, exclamations
    numeric         100 sentences   — dates, times, currency, counts
    pronunciation   100 sentences   — rare conjuncts, Sanskrit loanwords
    long            100 sentences   — 15+ word complex clauses

You provide the sentences; this module just structures, stores, and loads them.

File format (JSON): a simple list of {id, category, text} records.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
import hashlib
import json


VALID_CATEGORIES = ("narration", "dialogue", "numeric", "pronunciation", "long")


@dataclass
class TestItem:
    id: str
    category: str
    text: str

    def __post_init__(self) -> None:
        if self.category not in VALID_CATEGORIES:
            raise ValueError(
                f"category must be one of {VALID_CATEGORIES}, got {self.category!r}"
            )


def _make_id(text: str, category: str, index: int) -> str:
    h = hashlib.sha1(f"{category}:{text}".encode("utf-8")).hexdigest()[:8]
    return f"{category}_{index:04d}_{h}"


class TestSet:
    """Immutable-by-convention locked test set."""

    def __init__(self, items: list[TestItem]):
        self.items: list[TestItem] = list(items)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_category_dict(cls, per_category: dict[str, Iterable[str]]) -> "TestSet":
        """Build a test set from {"narration": [...], "dialogue": [...], ...}.

        Each sentence gets a deterministic ID derived from text + category.
        """
        items: list[TestItem] = []
        for category in VALID_CATEGORIES:
            sentences = list(per_category.get(category, []))
            for i, text in enumerate(sentences):
                text = text.strip()
                if not text:
                    continue
                items.append(TestItem(
                    id=_make_id(text, category, i),
                    category=category,
                    text=text,
                ))
        return cls(items)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"items": [asdict(it) for it in self.items]}
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "TestSet":
        data = json.loads(path.read_text(encoding="utf-8"))
        items = [TestItem(**d) for d in data["items"]]
        return cls(items)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def by_category(self, category: str) -> list[TestItem]:
        return [it for it in self.items if it.category == category]

    def count_by_category(self) -> dict[str, int]:
        out: dict[str, int] = {c: 0 for c in VALID_CATEGORIES}
        for it in self.items:
            out[it.category] += 1
        return out

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)
