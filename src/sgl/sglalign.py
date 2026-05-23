"""sgl.sglalign — Needleman-Wunsch 全局对齐（NumPy + 仿射 Gap 惩罚）"""

from __future__ import annotations

from typing import Any, Callable
import numpy as np

from sgl._types import AlignedPair, AlignmentResult, ShuffleGroup

SimilarityFunc = Callable[[Any, Any], float]

class Aligner:
    """Needleman-Wunsch 全局对齐器（仿射空隙惩罚）。

    参数:
        gap_open:          开启空隙惩罚（负值）
        gap_extend:        延续空隙惩罚（负值）
        mismatch_threshold: 判定不匹配的相似度阈值
    """

    def __init__(self, gap_open=-1.5, gap_extend=-0.2, mismatch_threshold=0.2):
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        self.mismatch_threshold = mismatch_threshold

    # ---- 核心对齐 ----

    def align(self, seq1: list[Any], seq2: list[Any],
              similarity_func: SimilarityFunc) -> AlignmentResult:
        n, m = len(seq1), len(seq2)
        g_o, g_e, th = self.gap_open, self.gap_extend, self.mismatch_threshold

        M = np.full((n + 1, m + 1), -np.inf, dtype=np.float64)
        X = np.full((n + 1, m + 1), -np.inf, dtype=np.float64)
        Y = np.full((n + 1, m + 1), -np.inf, dtype=np.float64)
        M[0, 0] = 0.0
        X[1:, 0] = g_o + np.arange(n, dtype=np.float64) * g_e
        Y[0, 1:] = g_o + np.arange(m, dtype=np.float64) * g_e

        for i in range(1, n + 1):
            si = seq1[i - 1]
            sim_row = np.array([similarity_func(si, sj) for sj in seq2], dtype=np.float64)
            scores = np.where(sim_row >= th, sim_row, g_o * 1.5)

            # M[i,j] = max(M[i-1,j-1], X[i-1,j-1], Y[i-1,j-1]) + score
            diag_max = np.maximum(np.maximum(M[i - 1, :-1], X[i - 1, :-1]), Y[i - 1, :-1])
            M[i, 1:] = diag_max + scores

            # X[i,j] = max(M[i-1,j] + g_o, X[i-1,j] + g_e)
            X[i, 1:] = np.maximum(M[i - 1, 1:] + g_o, X[i - 1, 1:] + g_e)

            # Y[i,j] = max(M[i,j-1] + g_o, Y[i,j-1] + g_e)  行内递推
            m_plus_open = M[i, :-1] + g_o
            yrow = np.empty(m + 1, dtype=np.float64)
            yrow[0] = Y[i, 0]
            for j in range(1, m + 1):
                cand = m_plus_open[j - 1]
                yrow[j] = cand if cand > yrow[j - 1] + g_e else yrow[j - 1] + g_e
            Y[i, :] = yrow

        combined = np.maximum.reduce([M, X, Y])
        total_score = float(combined[n, m])
        pairs, path = self._backtrace(seq1, seq2, M, X, Y, similarity_func)

        return AlignmentResult(
            pairs=pairs, score=total_score,
            n_source=n, n_target=m,
            source_seqs=list(seq1), target_seqs=list(seq2),
            dp_matrix=combined.tolist(), matrix_m=M.tolist(),
            matrix_x=X.tolist(), matrix_y=Y.tolist(),
            backtrace_path=path,
            gap_open=g_o, gap_extend=g_e, mismatch_threshold=th,
        )

    def alignfb(self, seq1: list[Any], seq2: list[Any],
                      similarity_func: SimilarityFunc) -> AlignmentResult:
        result = self.align(seq1, seq2, similarity_func)
        result.fb_matrix = self._compute_fb(seq1, seq2, similarity_func)
        return result

    # ---- 调换感知对齐 ----

    def reorderalign(self, seq1: list[Any], seq2: list[Any],
                            similarity_func: SimilarityFunc,
                            threshold: float = 0.3) -> AlignmentResult:
        """全局对齐，自动检测并处理句段顺序调换。

        算法：
        1. 计算完整 n×m 相似度矩阵
        2. 贪心匹配最佳 1:1 对（允许跨顺序匹配）
        3. 检测调换组：匹配的目标索引非单调的连续区域
        4. 输出结果，被调换的 pair 标记 is_shuffled=True
        """
        n, m = len(seq1), len(seq2)
        sim_mat = self._compute_sim_matrix(seq1, seq2, similarity_func)
        matches = self._greedy_match(sim_mat, threshold)
        matches.sort(key=lambda x: x[0])  # 按源索引排序
        tgt_seq = [p[1] for p in matches]
        shuffle_groups = self._detect_shuffle_groups(matches, tgt_seq)

        shuffled_src = {i for g in shuffle_groups for i in g.source_indices}
        match_map = {p[0]: (p[1], p[2]) for p in matches}
        matched_tgt = {p[1] for p in matches}

        pairs: list[AlignedPair] = []
        si = tj = 0
        while si < n or tj < m:
            if si < n and si in match_map:
                j, sim = match_map[si]
                # 插入该匹配之前未匹配的 target
                while tj < j:
                    if tj not in matched_tgt:
                        pairs.append(AlignedPair(None, seq2[tj], 0.0, True, "insert"))
                    tj += 1
                pairs.append(AlignedPair(
                    seq1[si], seq2[j], sim, False,
                    "match" if sim >= self.mismatch_threshold else "mismatch",
                    is_shuffled=si in shuffled_src,
                ))
                si += 1
                tj = max(tj, j + 1)
            elif si < n:
                pairs.append(AlignedPair(seq1[si], None, 0.0, True, "delete"))
                si += 1
            else:
                if tj not in matched_tgt:
                    pairs.append(AlignedPair(None, seq2[tj], 0.0, True, "insert"))
                tj += 1

        total_score = sum(p.similarity for p in pairs if not p.is_gap)

        return AlignmentResult(
            pairs=pairs, score=total_score,
            n_source=n, n_target=m,
            source_seqs=list(seq1), target_seqs=list(seq2),
            shuffle_groups=shuffle_groups or None,
            gap_open=self.gap_open, gap_extend=self.gap_extend,
            mismatch_threshold=self.mismatch_threshold,
        )

    @staticmethod
    def _compute_sim_matrix(seq1, seq2, sim_func) -> np.ndarray:
        n, m = len(seq1), len(seq2)
        mat = np.zeros((n, m))
        for i, s1 in enumerate(seq1):
            mat[i] = [sim_func(s1, s2) for s2 in seq2]
        return mat

    @staticmethod
    def _greedy_match(sim_mat: np.ndarray, threshold: float
                      ) -> list[tuple[int, int, float]]:
        """贪心寻找最佳 1:1 匹配（允许任意顺序）。"""
        n, m = sim_mat.shape
        flat = [(i, j, float(sim_mat[i, j])) for i in range(n) for j in range(m)
                if sim_mat[i, j] >= threshold]
        flat.sort(key=lambda x: -x[2])
        matches: list[tuple[int, int, float]] = []
        used_src: set[int] = set()
        used_tgt: set[int] = set()
        for i, j, s in flat:
            if i not in used_src and j not in used_tgt:
                matches.append((i, j, s))
                used_src.add(i)
                used_tgt.add(j)
        return matches

    @staticmethod
    def _detect_shuffle_groups(
        matches: list[tuple[int, int, float]], tgt_seq: list[int]
    ) -> list[ShuffleGroup]:
        """检测调换组：逐位置对比期望索引，找出被调换的连续匹配。"""
        if len(matches) < 2:
            return []
        expected = sorted(tgt_seq)
        # 找出 tgt 索引 ≠ 期望索引的位置
        bad = [k for k in range(len(matches)) if tgt_seq[k] != expected[k]]
        if len(bad) < 2:
            return []
        # 将连续的被调换位置合并为组
        groups: list[list[int]] = [[bad[0]]]
        for k in bad[1:]:
            if k == groups[-1][-1] + 1:
                groups[-1].append(k)
            else:
                groups.append([k])
        result: list[ShuffleGroup] = []
        for g in groups:
            if len(g) >= 2:
                result.append(ShuffleGroup(
                    source_indices=[matches[k][0] for k in g],
                    target_indices=[matches[k][1] for k in g],
                ))
        return result

    def _backtrace(self, seq1, seq2, M, X, Y, sim_func):
        n, m = len(seq1), len(seq2)
        combined = np.maximum.reduce([M, X, Y])
        pairs: list[AlignedPair] = []
        path: list[tuple[int, int, str]] = []

        i, j = n, m
        curr = 0  # 0=M, 1=X, 2=Y
        mx = float(combined[i, j])
        if np.isclose(mx, float(X[i, j])):
            curr = 1
        elif np.isclose(mx, float(Y[i, j])):
            curr = 2
        path.append((i, j, "MXY"[curr]))

        while i > 0 or j > 0:
            if curr == 0 and i > 0 and j > 0:
                sim = sim_func(seq1[i - 1], seq2[j - 1])
                sc = sim if sim >= self.mismatch_threshold else self.gap_open * 1.5
                prev = float(M[i, j]) - sc
                pairs.append(AlignedPair(
                    source=seq1[i - 1], target=seq2[j - 1],
                    similarity=sim, is_gap=False,
                    state="match" if sim >= self.mismatch_threshold else "mismatch",
                ))
                cand = [(0, float(M[i - 1, j - 1])),
                        (1, float(X[i - 1, j - 1])),
                        (2, float(Y[i - 1, j - 1]))]
                curr = max(cand, key=lambda t: t[1] if np.isclose(t[1], prev) else -np.inf)[0]
                i -= 1; j -= 1
            elif curr == 1 and i > 0:
                pairs.append(AlignedPair(source=seq1[i - 1], target=None, is_gap=True, state="delete"))
                prev = float(X[i, j])
                curr = 0 if np.isclose(prev, float(M[i - 1, j]) + self.gap_open) else 1
                i -= 1
            elif curr == 2 and j > 0:
                pairs.append(AlignedPair(source=None, target=seq2[j - 1], is_gap=True, state="insert"))
                prev = float(Y[i, j])
                curr = 0 if np.isclose(prev, float(M[i, j - 1]) + self.gap_open) else 2
                j -= 1
            else:
                break
            path.append((i, j, "MXY"[curr]))

        pairs.reverse()
        path.reverse()
        return pairs, path

    # ---- Forward-Backward ----

    def _compute_fb(self, seq1, seq2, sim_func):
        n, m = len(seq1), len(seq2)
        g_o, g_e, th = self.gap_open, self.gap_extend, self.mismatch_threshold

        fwd_dp = self._compute_fwd(seq1, seq2, sim_func)
        total = fwd_dp[n, m]

        BM = np.full((n + 1, m + 1), -np.inf, dtype=np.float64)
        BX = np.full((n + 1, m + 1), -np.inf, dtype=np.float64)
        BY = np.full((n + 1, m + 1), -np.inf, dtype=np.float64)
        BM[n, m] = BX[n, m] = BY[n, m] = 0.0
        for i in range(n - 1, -1, -1):
            BX[i, m] = g_o + (n - 1 - i) * g_e
        for j in range(m - 1, -1, -1):
            BY[n, j] = g_o + (m - 1 - j) * g_e

        for i in range(n - 1, -1, -1):
            si = seq1[i]
            sim_row = np.array([sim_func(si, sj) for sj in seq2], dtype=np.float64)
            scores = np.where(sim_row >= th, sim_row, g_o * 1.5)

            diag_max = np.maximum(np.maximum(BM[i + 1, 1:], BX[i + 1, 1:]), BY[i + 1, 1:])
            BM[i, :m] = diag_max + scores
            BX[i, :m] = np.maximum(BM[i + 1, :m] + g_o, BX[i + 1, :m] + g_e)

            b_m_open = BM[i, 1:] + g_o
            byrow = np.empty(m + 1, dtype=np.float64)
            byrow[m] = BY[i, m]
            for j in range(m - 1, -1, -1):
                byrow[j] = max(b_m_open[j], byrow[j + 1] + g_e)
            BY[i, :] = byrow

        b_combined = np.maximum.reduce([BM, BX, BY])
        return (fwd_dp + b_combined - total).tolist()  # type: ignore

    def _compute_fwd(self, seq1, seq2, sim_func):
        n, m = len(seq1), len(seq2)
        g_o, g_e, th = self.gap_open, self.gap_extend, self.mismatch_threshold

        M = np.full((n + 1, m + 1), -np.inf, dtype=np.float64)
        X = np.full((n + 1, m + 1), -np.inf, dtype=np.float64)
        Y = np.full((n + 1, m + 1), -np.inf, dtype=np.float64)
        M[0, 0] = 0.0
        X[1:, 0] = g_o + np.arange(n, dtype=np.float64) * g_e
        Y[0, 1:] = g_o + np.arange(m, dtype=np.float64) * g_e

        for i in range(1, n + 1):
            si = seq1[i - 1]
            sim_row = np.array([sim_func(si, sj) for sj in seq2], dtype=np.float64)
            scores = np.where(sim_row >= th, sim_row, g_o * 1.5)
            diag_max = np.maximum(np.maximum(M[i - 1, :-1], X[i - 1, :-1]), Y[i - 1, :-1])
            M[i, 1:] = diag_max + scores
            X[i, 1:] = np.maximum(M[i - 1, 1:] + g_o, X[i - 1, 1:] + g_e)
            m_open = M[i, :-1] + g_o
            yrow = np.empty(m + 1, dtype=np.float64)
            yrow[0] = Y[i, 0]
            for j in range(1, m + 1):
                yrow[j] = max(m_open[j - 1], yrow[j - 1] + g_e)
            Y[i, :] = yrow

        return np.maximum.reduce([M, X, Y])
