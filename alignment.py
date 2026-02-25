class NeedlemanWunschAligner:
    """
    实现 Needleman-Wunsch (NW) 全局对齐算法，支持仿射空隙惩罚 (Affine Gap Penalty)。
    用于句子级别或字符级别的对齐。
    """

    def __init__(self, gap_open=-1.5, gap_extend=-0.2, mismatch_threshold=0.2):
        """
        初始化对齐器。
        
        参数:
            gap_open (float): 开启空隙的惩罚值（调大以引入更严格的句段级惩罚）。
            gap_extend (float): 延续空隙的惩罚值。
            mismatch_threshold (float): 判定为不匹配的阈值（调大以过滤低信噪比匹配）。
        """
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        self.mismatch_threshold = mismatch_threshold

    def align(self, seq1, seq2, similarity_func):
        """
        使用仿射空隙惩罚进行全局对齐。
        """
        n, m = len(seq1), len(seq2)
        
        # M: Match/Mismatch 矩阵
        # X: 序列1中的 Gap (Delete)
        # Y: 序列2中的 Gap (Insert)
        M = [[-float('inf')] * (m + 1) for _ in range(n + 1)]
        X = [[-float('inf')] * (m + 1) for _ in range(n + 1)]
        Y = [[-float('inf')] * (m + 1) for _ in range(n + 1)]
        
        M[0][0] = 0
        for i in range(1, n + 1):
            X[i][0] = self.gap_open + (i - 1) * self.gap_extend
        for j in range(1, m + 1):
            Y[0][j] = self.gap_open + (j - 1) * self.gap_extend

        # 记录回溯路径
        # 为了支持 Forward-Backward，我们需要完整的得分矩阵
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                sim = similarity_func(seq1[i-1], seq2[j-1])
                score = sim if sim >= self.mismatch_threshold else (self.gap_open * 1.5)
                
                M[i][j] = max(M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1]) + score
                X[i][j] = max(M[i-1][j] + self.gap_open, X[i-1][j] + self.gap_extend)
                Y[i][j] = max(M[i][j-1] + self.gap_open, Y[i][j-1] + self.gap_extend)

        # 最终得分矩阵取三者最大值供可视化
        combined_dp = [[0.0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            for j in range(m + 1):
                combined_dp[i][j] = max(M[i][j], X[i][j], Y[i][j])

        self.last_dp_matrix = combined_dp
        self.last_M = M
        self.last_X = X
        self.last_Y = Y
        self.last_seq1 = seq1
        self.last_seq2 = seq2
        
        # 回溯
        alignment = []
        i, j = n, m
        curr_mat = 0 # 0: M, 1: X, 2: Y
        max_val = combined_dp[i][j]
        if max_val == X[i][j]: curr_mat = 1
        elif max_val == Y[i][j]: curr_mat = 2

        self.last_path = []

        while i > 0 or j > 0:
            self.last_path.append((i, j))
            if curr_mat == 0 and i > 0 and j > 0: # M
                alignment.append((seq1[i-1], seq2[j-1]))
                sim = similarity_func(seq1[i-1], seq2[j-1])
                score = sim if sim >= self.mismatch_threshold else (self.gap_open * 1.5)
                prev_val = M[i][j] - score
                
                if i > 1 and j > 1:
                    if abs(prev_val - M[i-1][j-1]) < 1e-9: curr_mat = 0
                    elif abs(prev_val - X[i-1][j-1]) < 1e-9: curr_mat = 1
                    elif abs(prev_val - Y[i-1][j-1]) < 1e-9: curr_mat = 2
                elif i > 1: # j=1, only M or X possible
                    if abs(prev_val - M[i-1][0]) < 1e-9: curr_mat = 0 # This case is tricky
                    else: curr_mat = 1
                elif j > 1: # i=1, only M or Y possible
                    if abs(prev_val - M[0][j-1]) < 1e-9: curr_mat = 0
                    else: curr_mat = 2
                else:
                    curr_mat = 0
                i -= 1
                j -= 1
            elif (curr_mat == 1 and i > 0) or (j == 0 and i > 0): # X (Delete)
                alignment.append((seq1[i-1], None))
                prev_val = X[i][j]
                if i > 1:
                    if abs(prev_val - (M[i-1][j] + self.gap_open)) < 1e-9: curr_mat = 0
                    elif abs(prev_val - (X[i-1][j] + self.gap_extend)) < 1e-9: curr_mat = 1
                    else: curr_mat = 1 # Fallback
                else:
                    curr_mat = 0
                i -= 1
            elif (curr_mat == 2 and j > 0) or (i == 0 and j > 0): # Y (Insert)
                alignment.append((None, seq2[j-1]))
                prev_val = Y[i][j]
                if j > 1:
                    if abs(prev_val - (M[i][j-1] + self.gap_open)) < 1e-9: curr_mat = 0
                    elif abs(prev_val - (Y[i][j-1] + self.gap_extend)) < 1e-9: curr_mat = 2
                    else: curr_mat = 2 # Fallback
                else:
                    curr_mat = 0
                j -= 1
            else:
                # 异常情况，强制终止
                break
        
        self.last_path.append((0, 0))
        alignment.reverse()
        return alignment

    def get_visualization_data(self, similarity_func):
        """
        获取用于可视化的矩阵数据，包含相似度矩阵和基于 Forward-Backward 思想的路径分值。
        """
        if not hasattr(self, 'last_dp_matrix'):
            return None
        
        n, m = len(self.last_seq1), len(self.last_seq2)
        
        # 1. 计算相似度矩阵 (Background)
        sim_matrix = [[0.0] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                sim_matrix[i][j] = similarity_func(self.last_seq1[i], self.last_seq2[j])
        
        # 2. 计算 Backward DP 矩阵 (同样使用仿射空隙)
        BM = [[-float('inf')] * (m + 1) for _ in range(n + 1)]
        BX = [[-float('inf')] * (m + 1) for _ in range(n + 1)]
        BY = [[-float('inf')] * (m + 1) for _ in range(n + 1)]
        
        BM[n][m] = 0
        for i in range(n - 1, -1, -1):
            BX[i][m] = self.gap_open + (n - 1 - i) * self.gap_extend
        for j in range(m - 1, -1, -1):
            BY[n][j] = self.gap_open + (m - 1 - j) * self.gap_extend
            
        for i in range(n - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                sim = similarity_func(self.last_seq1[i], self.last_seq2[j])
                score = sim if sim >= self.mismatch_threshold else (self.gap_open * 1.5)
                
                # Backward 转移
                BM[i][j] = max(BM[i+1][j+1], BX[i+1][j+1], BY[i+1][j+1]) + score
                BX[i][j] = max(BM[i+1][j] + self.gap_open, BX[i+1][j] + self.gap_extend)
                BY[i][j] = max(BM[i][j+1] + self.gap_open, BY[i][j+1] + self.gap_extend)
        
        # 3. 计算 Forward-Backward 组合矩阵
        fb_matrix = [[0.0] * (m + 1) for _ in range(n + 1)]
        total_score = self.last_dp_matrix[n][m]
        
        for i in range(n + 1):
            for j in range(m + 1):
                # 合并 Backward 三个矩阵的最大值
                b_max = max(BM[i][j], BX[i][j], BY[i][j])
                score_diff = self.last_dp_matrix[i][j] + b_max - total_score
                fb_matrix[i][j] = score_diff
                
        return {
            "matrix": fb_matrix,
            "dp_matrix": self.last_dp_matrix,
            "sim_matrix": sim_matrix,
            "seq1": self.last_seq1,
            "seq2": self.last_seq2,
            "path": getattr(self, 'last_path', []),
            "total_score": total_score
        }
