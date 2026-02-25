from alignment import NeedlemanWunschAligner

def char_similarity(c1, c2):
    """
    极简的字符相似度函数。
    """
    return 1.0 if c1 == c2 else 0.0

class CharLevelAligner:
    """
    处理句子内部的字符级差异对齐。
    """
    def __init__(self):
        # 字符级对齐通常需要更严厉的 Gap 惩罚，以防止过度破碎
        # 字符级对齐采用线性惩罚即可
        self.aligner = NeedlemanWunschAligner(gap_open=-0.8, gap_extend=-0.8, mismatch_threshold=0.1)

    def align_chars(self, text1, text2):
        """
        对两个句子进行字符级对齐。
        返回结果格式：[(char1, char2, type), ...]
        实现了“差异块合并”逻辑：连续的非等值字符会被聚合在一起，先排删除，后排新增。
        """
        if not text1:
            return [(None, c, 'insert') for c in text2]
        if not text2:
            return [(c, None, 'delete') for c in text1]
            
        raw_alignment = self.aligner.align(list(text1), list(text2), char_similarity)
        
        refined_alignment = []
        i = 0
        n = len(raw_alignment)
        
        while i < n:
            c1, c2 = raw_alignment[i]
            
            # 1. 如果是相等的字符，直接作为锚点
            if c1 and c2 and c1 == c2:
                refined_alignment.append((c1, c2, 'equal'))
                i += 1
                continue
            
            # 2. 如果是差异（删除、新增或替换），收集连续的差异块
            cluster_orig = []
            cluster_rew = []
            
            while i < n:
                curr_c1, curr_c2 = raw_alignment[i]
                # 遇到相等的字符，停止当前差异块的收集
                if curr_c1 and curr_c2 and curr_c1 == curr_c2:
                    break
                
                if curr_c1: cluster_orig.append(curr_c1)
                if curr_c2: cluster_rew.append(curr_c2)
                i += 1
            
            # 3. 将收集到的差异块输出：先全删，后全增
            # 这样在前端展示时，会呈现为 [删掉的一串][新增的一串]，而不是切碎的 [删增][删增]
            for c in cluster_orig:
                refined_alignment.append((c, None, 'delete'))
            for c in cluster_rew:
                refined_alignment.append((None, c, 'insert'))
                
        return refined_alignment

def get_rich_diff(alignment_result, similarity_func=None):
    """
    将句子级对齐结果转化为包含字符级差异的富文本格式数据。
    实现了“块合并”逻辑：将相邻的非等值对齐项合并为一个块进行字符级比对。
    """
    char_aligner = CharLevelAligner()
    rich_results = []
    
    # 降低锚点识别阈值，从严格的 == 改为相似度阈值
    ANCHOR_THRESHOLD = 0.80
    
    def is_anchor(s1, s2):
        if not s1 or not s2:
            return False
        if s1 == s2:
            return True
        if similarity_func:
            return similarity_func(s1, s2) >= ANCHOR_THRESHOLD
        return False

    i = 0
    n = len(alignment_result)
    
    while i < n:
        orig, rew = alignment_result[i]
        
        # 1. 处理匹配的块 (锚点)
        if is_anchor(orig, rew):
            if orig == rew:
                rich_results.append({
                    "type": "equal",
                    "orig": orig,
                    "rew": rew,
                    "chars": [(c, c, 'equal') for c in orig] 
                })
            else:
                # 虽然被识别为锚点（相似度高），但存在微小差异，执行字符级对齐
                char_diffs = char_aligner.align_chars(orig, rew)
                rich_results.append({
                    "type": "modify",
                    "orig": orig,
                    "rew": rew,
                    "chars": char_diffs
                })
            i += 1
            continue
            
        # 2. 发现差异，开始收集“不稳定块” (Cluster)
        cluster_orig_list = []
        cluster_rew_list = []
        
        while i < n:
            curr_orig, curr_rew = alignment_result[i]
            # 如果遇到匹配的句子，停止合并
            if is_anchor(curr_orig, curr_rew):
                break
            
            if curr_orig: cluster_orig_list.append(curr_orig)
            if curr_rew: cluster_rew_list.append(curr_rew)
            i += 1
            
        # 3. 对合并后的块进行字符级比对
        full_orig = "".join(cluster_orig_list)
        full_rew = "".join(cluster_rew_list)
        
        if full_orig and full_rew:
            # 这是一个复杂的修改块
            char_diffs = char_aligner.align_chars(full_orig, full_rew)
            rich_results.append({
                "type": "modify",
                "orig": full_orig,
                "rew": full_rew,
                "chars": char_diffs
            })
        elif full_orig:
            # 纯删除块
            rich_results.append({
                "type": "delete",
                "orig": full_orig,
                "rew": None,
                "chars": [(c, None, 'delete') for c in full_orig]
            })
        elif full_rew:
            # 纯新增块
            rich_results.append({
                "type": "insert",
                "orig": None,
                "rew": full_rew,
                "chars": [(None, c, 'insert') for c in full_rew]
            })
            
    return rich_results
