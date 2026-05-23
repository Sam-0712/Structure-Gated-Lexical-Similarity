"""sgl.sgldiff — 字符级差异对齐"""

from __future__ import annotations

from typing import Any, Callable, Optional

from sgl.sglalign import Aligner
from sgl._types import AlignedPair, CharDiff, DiffBlock, RichDiffResult

SimilarityFunc = Callable[[Any, Any], float]


def _char_sim(c1: str, c2: str) -> float:
    return 1.0 if c1 == c2 else 0.0


class CharLevelAligner:
    """字符级差异对齐（Gap 惩罚更严厉，差异块合并避免交错）。"""

    def __init__(self):
        self.aligner = Aligner(gap_open=-0.8, gap_extend=-0.8, mismatch_threshold=0.1)

    def align_chars(self, text1: str, text2: str) -> list[CharDiff]:
        if not text1:
            return [CharDiff(None, c, "insert") for c in text2]
        if not text2:
            return [CharDiff(c, None, "delete") for c in text1]

        raw = self.aligner.align(list(text1), list(text2), _char_sim).pairs
        refined: list[CharDiff] = []
        i, n = 0, len(raw)

        while i < n:
            p = raw[i]
            # 相等字符直接通过
            if p.source and p.target and p.source == p.target:
                refined.append(CharDiff(p.source, p.target, "equal"))
                i += 1
                continue

            # 差异块：连续不等字符先删后增
            orig, rew = [], []
            while i < n:
                cp = raw[i]
                if cp.source and cp.target and cp.source == cp.target:
                    break
                if cp.source: orig.append(cp.source)
                if cp.target: rew.append(cp.target)
                i += 1
            for c in orig:
                refined.append(CharDiff(c, None, "delete"))
            for c in rew:
                refined.append(CharDiff(None, c, "insert"))

        return refined


_ANCHOR = 0.80


def richdiff(
    alignment_result: list[AlignedPair],
    similarity_func: Optional[SimilarityFunc] = None,
    anchor_threshold: float = _ANCHOR,
) -> RichDiffResult:
    ca = CharLevelAligner()

    def anchor(s1: Optional[str], s2: Optional[str]) -> bool:
        if not s1 or not s2:
            return False
        return s1 == s2 or (similarity_func is not None and similarity_func(s1, s2) >= anchor_threshold)

    blocks: list[DiffBlock] = []
    i, n = 0, len(alignment_result)

    while i < n:
        p = alignment_result[i]
        if anchor(p.source, p.target):
            if p.source == p.target:
                blocks.append(DiffBlock("equal", p.source, p.target, [CharDiff(c, c, "equal") for c in p.source]))
            else:
                blocks.append(DiffBlock("modify", p.source, p.target, ca.align_chars(p.source or "", p.target or "")))
            i += 1
            continue

        orig, rew = [], []
        while i < n:
            cp = alignment_result[i]
            if anchor(cp.source, cp.target):
                break
            if cp.source: orig.append(cp.source)
            if cp.target: rew.append(cp.target)
            i += 1

        fo, fr = "".join(orig), "".join(rew)
        if fo and fr:
            blocks.append(DiffBlock("modify", fo, fr, ca.align_chars(fo, fr)))
        elif fo:
            blocks.append(DiffBlock("delete", fo, None, [CharDiff(c, None, "delete") for c in fo]))
        elif fr:
            blocks.append(DiffBlock("insert", None, fr, [CharDiff(None, c, "insert") for c in fr]))

    return RichDiffResult(blocks=blocks)


def chardiff(text1: str, text2: str) -> list[CharDiff]:
    return CharLevelAligner().align_chars(text1, text2)
