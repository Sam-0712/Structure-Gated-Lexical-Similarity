"""sgl._types — 结构化输出类型"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from typing import Any, Optional


def _to_native(val: Any) -> Any:
    import numpy as np
    if isinstance(val, np.floating):
        return float(val)
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def _deep_native(obj: Any, round4: bool = True) -> Any:
    if isinstance(obj, dict):
        return {k: _deep_native(v, round4) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deep_native(v, round4) for v in obj]
    val = _to_native(obj)
    if round4 and isinstance(val, float) and math.isfinite(val):
        return round(val, 4)
    return val


# ============================================================

@dataclass
class SimilarityResult:
    score: float
    dice_score: float
    lcs_score: float
    roc_score: float
    complexity_a: float
    complexity_b: float
    n_gram_order: int
    n_gram_weights: dict[int, float]
    logic_reward: float
    len_penalty: float
    dispersion_consistency: float
    struct_ratio: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return _deep_native(asdict(self))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def summary(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "dice": round(self.dice_score, 4),
            "lcs": round(self.lcs_score, 4),
            "roc": round(self.roc_score, 4),
            "reward": round(self.logic_reward, 4),
        }


@dataclass
class ShuffleGroup:
    """一组被打乱顺序的句子对。"""
    source_indices: list[int]
    target_indices: list[int]


@dataclass
class AlignedPair:
    source: Optional[str]
    target: Optional[str]
    similarity: float = 0.0
    is_gap: bool = False
    state: str = "match"
    is_shuffled: bool = False


@dataclass
class AlignmentResult:
    pairs: list[AlignedPair]
    score: float
    n_source: int
    n_target: int
    source_seqs: list[str]
    target_seqs: list[str]
    dp_matrix: Optional[list[list[float]]] = None
    matrix_m: Optional[list[list[float]]] = None
    matrix_x: Optional[list[list[float]]] = None
    matrix_y: Optional[list[list[float]]] = None
    backtrace_path: Optional[list[tuple[int, int, str]]] = None
    fb_matrix: Optional[list[list[float]]] = None
    shuffle_groups: Optional[list[ShuffleGroup]] = None
    gap_open: float = -1.5
    gap_extend: float = -0.2
    mismatch_threshold: float = 0.2

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "n_source": self.n_source,
            "n_target": self.n_target,
            "score": round(self.score, 4),
            "pairs": [
                {"source": p.source, "target": p.target,
                 "similarity": round(p.similarity, 4), "is_gap": p.is_gap,
                 "state": p.state, "is_shuffled": p.is_shuffled}
                for p in self.pairs
            ],
            "backtrace_path": self.backtrace_path,
            "gap_open": round(self.gap_open, 4),
            "gap_extend": round(self.gap_extend, 4),
            "mismatch_threshold": round(self.mismatch_threshold, 4),
        }
        for k in ("dp_matrix", "matrix_m", "matrix_x", "matrix_y", "fb_matrix"):
            v = getattr(self, k)
            if v is not None:
                d[k] = _deep_native(v)
        if self.shuffle_groups:
            d["shuffle_groups"] = [
                {"source_indices": g.source_indices, "target_indices": g.target_indices}
                for g in self.shuffle_groups
            ]
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {"match": 0, "mismatch": 0, "delete": 0, "insert": 0}
        shuffled = 0
        for p in self.pairs:
            counts[p.state] += 1
            if p.is_shuffled:
                shuffled += 1
        result = {"score": round(self.score, 4), f"{self.n_source}x{self.n_target}": "align", **counts}
        if shuffled:
            result["shuffled"] = shuffled
        return result


@dataclass
class CharDiff:
    char_src: Optional[str]
    char_tgt: Optional[str]
    diff_type: str


@dataclass
class DiffBlock:
    block_type: str
    source_text: Optional[str]
    target_text: Optional[str]
    chars: list[CharDiff]


@dataclass
class RichDiffResult:
    blocks: list[DiffBlock]
    has_char_matrix: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _deep_native(asdict(self))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {"equal": 0, "modify": 0, "delete": 0, "insert": 0}
        for b in self.blocks:
            counts[b.block_type] += 1
        return counts

    def to_diff_text(self) -> str:
        lines: list[str] = []
        for b in self.blocks:
            if b.block_type == "equal":
                lines.append(f"  {b.source_text}")
            elif b.block_type == "modify":
                lines.append(f"- {b.source_text}")
                lines.append(f"+ {b.target_text}")
            elif b.block_type == "delete":
                lines.append(f"- {b.source_text}")
            elif b.block_type == "insert":
                lines.append(f"+ {b.target_text}")
        return "\n".join(lines)
