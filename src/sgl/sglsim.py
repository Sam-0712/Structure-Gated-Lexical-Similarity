"""sgl.sglsim — 混合相似度计算 (V4)"""

from functools import lru_cache
import math
import numpy as np
from sgl._types import SimilarityResult


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0.0 else 0.0


def get_text_complexity(s: str) -> float:
    return len(set(s)) / len(s) if len(s) > 0 else 0.0


def dice_coefficient_with_pos(x: str, y: str, n: int) -> tuple[float, list[int], list[int]]:
    gx = [x[i:i + n] for i in range(len(x) - n + 1)] if len(x) >= n else []
    gy = [y[i:i + n] for i in range(len(y) - n + 1)] if len(y) >= n else []

    cx: dict[str, list[int]] = {}
    for i, g in enumerate(gx):
        cx.setdefault(g, []).append(i)
    cy: dict[str, list[int]] = {}
    for i, g in enumerate(gy):
        cy.setdefault(g, []).append(i)

    overlap = 0
    px: list[int] = []
    py: list[int] = []
    for g in cx:
        if g in cy:
            c = min(len(cx[g]), len(cy[g]))
            overlap += c
            px.extend(cx[g][:c])
            py.extend(cy[g][:c])

    return safe_div(2.0 * overlap, len(gx) + len(gy)), px, py


def get_lis_length(arr: list[int]) -> int:
    tails: list[int] = []
    for num in arr:
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] < num:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(tails):
            tails.append(num)
        else:
            tails[lo] = num
    return len(tails)


def compute_roc(x: str, y: str) -> float:
    y_pos: dict[str, list[int]] = {}
    for i, ch in enumerate(y):
        y_pos.setdefault(ch, []).append(i)

    indices: list[int] = []
    ptr = {ch: 0 for ch in y_pos}
    common = 0
    for ch in x:
        if ch in y_pos and ptr[ch] < len(y_pos[ch]):
            indices.append(y_pos[ch][ptr[ch]])
            ptr[ch] += 1
            common += 1
    if common == 0:
        return 0.0
    return get_lis_length(indices) / common


def lcs_ratio(x: str, y: str) -> float:
    n, m = len(x), len(y)
    if n * m == 0:
        return 0.0
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        xi = x[i - 1]
        for j in range(1, m + 1):
            cur[j] = prev[j - 1] + 1 if xi == y[j - 1] else (prev[j] if prev[j] > cur[j - 1] else cur[j - 1])
        prev = cur
    return (2.0 * prev[m]) / (n + m)


def compute_dispersion(positions: list[int], text_length: int) -> float:
    if len(positions) <= 1:
        return 0.5
    std = float(np.std(positions, ddof=0))
    return min(std / (text_length / 2.0 + 1e-9), 1.0)


def compute_weights(text_length: int, n_gram_order: int) -> list[float]:
    if n_gram_order <= 1:
        return [1.0]
    w = [1.0 / (k * math.log(k + 1)) for k in range(1, n_gram_order + 1)]
    t = sum(w)
    return [v / t for v in w] if t > 0 else [1.0 / n_gram_order] * n_gram_order


@lru_cache(maxsize=4096)
def _cached_full(x: str, y: str) -> tuple[float, ...]:
    return _compute_full(x, y)


def _compute_full(x: str, y: str) -> tuple:
    if not x or not y:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0, 0.0, 0.0)

    len_min = min(len(x), len(y))
    len_max = max(len(x), len(y))
    alphabet = len(set(x) | set(y))
    complexity = (get_text_complexity(x) + get_text_complexity(y)) / 2.0

    n_gram_order = max(1, len_min.bit_length() - 1)
    weights = compute_weights(len_min, n_gram_order)

    dice_sum = 0.0
    all_px: list[int] = []
    all_py: list[int] = []
    match_count = 0
    log_len = (math.log(len(x) + 1) + math.log(len(y) + 1)) / 2.0 + 1.0

    for i, order in enumerate(range(1, n_gram_order + 1)):
        ds, px, py = dice_coefficient_with_pos(x, y, order)
        dice_sum += weights[i] * ds
        if order == 1:
            match_count = len(px)
        if order <= log_len:
            all_px.extend(px)
            all_py.extend(py)

    lcs_score = lcs_ratio(x, y)
    roc_score = compute_roc(x, y)

    ln_len_min = math.log(len_min + 1)
    ln_context = math.log(len_max + alphabet + 1)
    struct_ratio = safe_div(ln_len_min, ln_context) * complexity
    base_score = dice_sum * (1.0 - struct_ratio) + lcs_score * struct_ratio

    if all_px and all_py:
        dx = compute_dispersion(all_px, len(x))
        dy = compute_dispersion(all_py, len(y))
        disp_consistency = 1.0 - abs(dx - dy)
    else:
        disp_consistency = 0.0

    expected_random = 1.0 / (alphabet + 1.0)
    confidence = safe_div(match_count / len_min - expected_random, 1.0 - expected_random)
    disp_weight = max(0.0, confidence) * complexity
    logic_reward = roc_score ** (1.0 / (complexity + 1e-9))
    gamma = safe_div(ln_len_min, math.log(len_max + 1))
    len_penalty = (len_min / len_max) ** gamma

    final = len_penalty * logic_reward * (base_score + disp_consistency * disp_weight) / (1.0 + disp_weight)

    return (final, dice_sum, lcs_score, roc_score, complexity, complexity,
            n_gram_order, struct_ratio, logic_reward, len_penalty, disp_consistency, confidence)


def hybrid_similarity(x: str, y: str) -> float:
    return round(float(_cached_full(x, y)[0]), 4)


def hybriddetail(x: str, y: str) -> SimilarityResult:
    r = _compute_full(x, y)
    cpx, cpy = get_text_complexity(x), get_text_complexity(y)
    order = r[6]
    w = compute_weights(min(len(x), len(y)), order)
    return SimilarityResult(
        score=round(r[0], 4), dice_score=round(r[1], 4),
        lcs_score=round(r[2], 4), roc_score=round(r[3], 4),
        complexity_a=round(cpx, 4), complexity_b=round(cpy, 4),
        n_gram_order=order,
        n_gram_weights={k + 1: round(v, 4) for k, v in enumerate(w)},
        logic_reward=round(r[8], 4), len_penalty=round(r[9], 4),
        dispersion_consistency=round(r[10], 4),
        struct_ratio=round(r[7], 4), confidence=round(r[11], 4),
    )


def cacheclear() -> None:
    _cached_full.cache_clear()


def cacheinfo():
    return _cached_full.cache_info()
