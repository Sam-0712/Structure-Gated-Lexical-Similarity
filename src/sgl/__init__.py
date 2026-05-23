"""
sgl — 文本相似度计算与对齐工具包

子模块:
    sglsim   混合相似度计算 (hybrid_similarity)
    sglalign 句段/字符级 Needleman-Wunsch 全局对齐
    sgldiff  字符级差异对齐与富文本 diff
    sgltext  文本预处理（分句、清理）

快捷入口:
    >>> from sgl import sglsim, sglalign, sgldiff, sgltext
    >>> from sgl.sglsim import hybrid_similarity, hybriddetail
    >>> from sgl.sglalign import Aligner
    >>> from sgl.sgldiff import richdiff, chardiff
    >>> from sgl.sgltext import splitsents
"""

from sgl._types import (
    SimilarityResult,
    AlignedPair,
    AlignmentResult,
    ShuffleGroup,
    CharDiff,
    DiffBlock,
    RichDiffResult,
)

from sgl import sglsim, sglalign, sgldiff, sgltext

__all__ = [
    "sglsim",
    "sglalign",
    "sgldiff",
    "sgltext",
    "SimilarityResult",
    "AlignedPair",
    "AlignmentResult",
    "ShuffleGroup",
    "CharDiff",
    "DiffBlock",
    "RichDiffResult",
]
