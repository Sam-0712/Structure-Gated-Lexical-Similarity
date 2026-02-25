"""
文本相似度计算算法 V4
包含 N-gram、顺序一致性、离散度、动态权重等机制
"""

# ------------------------------------------------------------
# 工具函数与常量
# ------------------------------------------------------------

def safe_div(a: float, b: float) -> float:
    """安全除法，避免零除错误"""
    return a / b if b != 0.0 else 0.0

def approx_ln(x: float) -> float:
    """
    使用启发式方法近似计算 ln(x)。
    利用 bit_length() 快速获取 log2，再换算为 ln。
    """
    if x <= 0: return -10.0
    if x < 1.0: return -approx_ln(1.0 / x)
    
    # 用近似方法计算 ln(2)
    def approx_ln2():
        result = 0.0
        term = 1.0
        n = 1
        for i in range(1, 5):
            term *= 0.5
            result += term / i
        return result
    
    # 计算 log2(x) 并转换为 ln(x)
    log2_approx = float(int(x).bit_length()) - 0.5
    ln2 = approx_ln2()

    return log2_approx * ln2

def get_text_complexity(s: str) -> float:
    """计算文本复杂度（字符多样性因子）"""
    return len(set(s)) / len(s) if len(s) > 0 else 0.0

# ------------------------------------------------------------
# N-gram 相关
# ------------------------------------------------------------

def dice_coefficient_with_pos(x: str, y: str, n: int) -> tuple[float, list[int], list[int]]:
    """
    计算带位置信息的 Dice 系数
    返回: (dice_score, positions_in_x, positions_in_y)
    """
    # 内联 n-gram 提取
    gx = [x[i:i+n] for i in range(len(x) - n + 1)] if len(x) >= n else []
    gy = [y[i:i+n] for i in range(len(y) - n + 1)] if len(y) >= n else []

    # 记录每个 n-gram 的位置
    cx = {}
    for i, g in enumerate(gx):
        cx.setdefault(g, []).append(i)
    cy = {}
    for i, g in enumerate(gy):
        cy.setdefault(g, []).append(i)

    # 计算匹配数量和对应位置
    overlap = 0
    px, py = [], []
    for g in cx:
        if g in cy:
            count = min(len(cx[g]), len(cy[g]))
            overlap += count
            px.extend(cx[g][:count])
            py.extend(cy[g][:count])

    score = safe_div(2.0 * overlap, len(gx) + len(gy))
    return score, px, py

# ------------------------------------------------------------
# 顺序一致性与结构分析
# ------------------------------------------------------------

def get_lis_length(arr: list[int]) -> int:
    """
    使用二分查找计算最长递增子序列长度 (O(n log n))
    """
    tails = [] # tails[i] 表示长度为 i+1 的递增子序列的最小末尾值
    for num in arr:
        # 使用二分查找找到 num 在 tails 中的插入位置
        left, right = 0, len(tails)
        
        while left < right:
            mid = (left + right) // 2
            if tails[mid] < num:
                left = mid + 1
            else:
                right = mid
        
        # 如果找到了合适的位置，更新该位置的尾部值
        if left == len(tails):
            tails.append(num)      # 扩展最长子序列
        else:
            tails[left] = num      # 优化现有子序列的尾部值
    
    return len(tails)

def compute_roc(x: str, y: str) -> float:
    """
    计算相对顺序一致性 (Relative Order Consistency)
    衡量 x 中的公共字符在 y 中是否保持了相对顺序
    """
    # 构建 y 中每个字符的位置列表
    y_positions = {}
    for i, char in enumerate(y):
        y_positions.setdefault(char, []).append(i)
    
    # 按 x 中字符顺序，在 y 中找到对应位置
    indices = []
    y_ptr = {char: 0 for char in y_positions}
    common_count = 0
    
    for char in x:
        if char in y_positions and y_ptr[char] < len(y_positions[char]):
            indices.append(y_positions[char][y_ptr[char]])
            y_ptr[char] += 1
            common_count += 1
            
    if common_count == 0:
        return 0.0
    
    lis_len = get_lis_length(indices)
    return lis_len / common_count

def lcs_ratio(x: str, y: str) -> float:
    """
    计算最长公共子序列 (LCS) 的比例，返回 2 * LCS / (|x| + |y|)，与 Dice 系数保持量纲一致
    """
    n, m = len(x), len(y)
    
    # 空间优化：使用一维 DP 数组
    prev_dp = [0] * (m + 1)
    for i in range(1, n + 1):
        curr_dp = [0] * (m + 1)
        for j in range(1, m + 1):
            if x[i-1] == y[j-1]:
                curr_dp[j] = prev_dp[j-1] + 1
            else:
                curr_dp[j] = max(prev_dp[j], curr_dp[j-1])
        prev_dp = curr_dp
            
    return (2.0 * prev_dp[m]) / (n + m) if m * n > 0.0 else 0.0

# ------------------------------------------------------------
# 离散度与权重计算
# ------------------------------------------------------------

def compute_dispersion(positions: list[int], text_length: int) -> float:
    """
    计算匹配位置的离散度
    使用位置的标准差归一化
    """
    avg_pos = sum(positions) / len(positions)
    variance = sum((p - avg_pos) ** 2 for p in positions) / len(positions)
    std_dev = variance ** 0.5
    
    # 理论最大标准差（两端分布）
    max_std = text_length / 2.0
    return min(std_dev / (max_std + 1e-9), 1.0) if len(positions) > 1 else 0.5

def compute_weights(text_length: int, n_gram_order: int) -> list[float]:
    """
    计算基于齐普夫定律 (Zipf's Law) 的 N-gram 权重分布
    """
    if n_gram_order <= 1:
        return [1.0]
    
    # 计算每个阶数的权重（使用近似对数）
    weights = []
    for k in range(1, n_gram_order + 1):
        weight = 1.0 / (k * approx_ln(float(k + 1)))
        weights.append(weight)
    
    # 归一化
    total_weight = sum(weights)
    return [w / total_weight for w in weights] if total_weight > 0 else [1.0 / n_gram_order] * n_gram_order

# ------------------------------------------------------------
# 主算法：混合相似度计算
# ------------------------------------------------------------

def hybrid_similarity(x: str, y: str) -> float:
    """
    计算两个字符串的混合相似度（顺序一致性奖励版 V4）
    
    特点：
    1. 引入 ROC (Relative Order Consistency) 衡量逻辑顺序，奖励“意义保持”
    2. 简化数学模型，使用启发式对数近似
    3. 动态平衡：根据匹配质量和顺序一致性自动分配权重
    """
    # 处理空字符串
    if not x or not y:
        return 0.0
        
    # 基础统计
    len_min, len_max, len_avg = min(len(x), len(y)), max(len(x), len(y)), (len(x) + len(y)) / 2.0
    alphabet_size = len(set(x) | set(y))
    complexity = (get_text_complexity(x) + get_text_complexity(y)) / 2.0

    # 动态确定最大 N-gram 阶数
    n_gram_order = max(1, len_min.bit_length() - 1)

    # 计算各阶 N-gram 的加权 Dice 分数和位置信息
    weights = compute_weights(len_min, n_gram_order)
    dice_sum = 0.0
    all_px, all_py = [], []  # 用于离散度计算
    match_count = 0  # 用于置信度计算
    
    for i, order in enumerate(range(1, n_gram_order + 1)):
        dice_score, px, py = dice_coefficient_with_pos(x, y, order)
        dice_sum += weights[i] * dice_score
        if order == 1:
            match_count = len(px)  # 一元 gram 的匹配数用于置信度
        if order <= (approx_ln(len(x)) + approx_ln(len(y))) / 2 + 1: # 位置信息计算离散度
            all_px.extend(px)
            all_py.extend(py)

    # 结构分析：LCS 和 顺序一致性
    lcs_score = lcs_ratio(x, y)
    roc_score = compute_roc(x, y)

    # 动态权重：结构与词法的平衡
    ln_len_min = approx_ln(float(len_min + 1))
    ln_context = approx_ln(float(len_max + alphabet_size + 1))
    struct_ratio = safe_div(ln_len_min, ln_context) * complexity
    base_score = dice_sum * (1.0 - struct_ratio) + lcs_score * struct_ratio

    # 离散度一致性评估
    if all_px and all_py:
        disp_x = compute_dispersion(all_px, len(x))
        disp_y = compute_dispersion(all_py, len(y))
        disp_consistency = 1.0 - abs(disp_x - disp_y)
    else:
        disp_consistency = 0.0

    # 离散度权重：由匹配置信度和文本复杂度决定
    expected_random_match = 1.0 / (alphabet_size + 1.0)
    confidence = safe_div(match_count / len_min - expected_random_match, 1.0 - expected_random_match)
    disp_weight = max(0.0, confidence) * complexity

    # 顺序奖励：ROC 作为逻辑连贯性的门控
    logic_reward = roc_score ** (1.0 / (complexity + 1e-9))

    # 长度惩罚：阻尼效应
    gamma = safe_div(ln_len_min, approx_ln(float(len_max + 1)))
    len_penalty = (len_min / len_max) ** gamma

    # 最终分数：长度惩罚 * 逻辑奖励 * (基础分数 + 离散度一致性 * 离散度权重) / (1 + 离散度权重)
    final_score = len_penalty * logic_reward * (base_score + disp_consistency * disp_weight) / (1.0 + disp_weight)

    return final_score # 无任何约束，仅仅依靠函数自然计算；分数约束min(max(final_score, 0.0), 1.0) 

# ------------------------------------------------------------
# 测试用例
# ------------------------------------------------------------

if __name__ == "__main__":
    import time

    test_cases = [
        # 空字符串检测
        ("", ""),

        # 完全相同的短文本
        ("你好世界", "你好世界"),
        
        # 包含关系的短文本
        ("你好世界", "你好"),
        ("人工智能", "人工"),
        ("哈哈哈", "哈哈"),
        
        # 词序变换的短文本
        ("你好世界", "世界你好"),
        ("马到成功", "成功到马"),
        ("春夏秋冬", "冬秋夏春"),
        
        # 语义相关的短文本
        ("算法设计", "算法实现"),
        ("电脑编程", "计算机编码"),
        ("学生学习", "教育培养"),
        
        # 语义相似但表达不同的文本
        ("秋天非常美丽", "秋色极美"),
        ("今天天气很好", "今日气候宜人"),
        ("我非常高兴", "我十分开心"),
        
        # 完全不相关的短文本
        ("完全不同", "毫不相关"),
        ("苹果手机", "数学公式"),
        ("游泳健身", "历史文学"),
        
        # 较长文本 - 相似内容
        ("深度学习是机器学习的一个分支", "深度学习属于机器学习领域"),
        ("清华大学是中国著名的高等学府", "清华是中国知名大学"),
        
        # 较长文本 - 部分重叠
        ("今天我去超市买了苹果和香蕉", "昨天我买了香蕉和橙子"),
        ("北京是中国的首都城市", "上海是中国最大的城市"),
        
        # 较长文本 - 完全不同
        ("量子力学是物理学的分支学科", "红楼梦是中国古典文学名著"),
        ("太阳能是一种可再生能源", "太极拳是传统武术的一种"),
        
        # 复杂句式 - 相似含义
        ("尽管天气不好，但我们还是如期举行了活动", "虽然气候不佳，但活动照常进行"),
        ("由于疫情原因，会议改为线上举办", "鉴于疫情影响，会议转为在线举行"),
        
        # 包含数字和专有名词
        ("2023年世界杯在北京举行", "2023年世界杯于北京举办"),
        ("iPhone14发布了新功能", "苹果iPhone14推出新特性"),
        
        # 同义词替换
        ("快速跑步", "迅速奔跑"),
        ("美丽风景", "漂亮景色"),
        
        # 反义词对比
        ("白天很长", "夜晚很短"),
        ("价格很高", "成本低廉"),
        
        # 添加修饰词
        ("一只小猫", "一只可爱的小花猫"),
        ("读书学习", "认真读书学习")
    ]

    print("文本相似度测试结果：")
    print("-" * 80)
    
    for text_a, text_b in test_cases:
        start_time = time.perf_counter()
        similarity = hybrid_similarity(text_a, text_b)
        elapsed_time = (time.perf_counter() - start_time) * 1_000_000  # 微秒
        print(f"（A = “{text_a}”，B = “{text_b}”，相似度：{similarity:.4f}，耗时：{elapsed_time:.2f}μs）")