# Structure-Gated Lexical Similarity

A lightweight, dependency-free text alignment framework for Chinese essay correction. It operates in low-resource environments without external computing libraries, implementing a hierarchical approach: sentence-level similarity matching, global sequence alignment via Needleman-Wunsch with affine gap penalties, and character-level refinement with difference block merging.

## Algorithm Architecture

The framework solves the problem of aligning LLM-revised essays with original texts through a three-layer architecture designed for both structural stability and granular precision.

### 1. Sentence-Level Similarity: Hybrid Dice Coefficient

Instead of a single metric, the similarity function $\mathcal{F}(x, y)$ synthesizes multiple signals to distinguish genuine revisions from unrelated text:

- **Multi-order N-gram Dice ($S_{dice}$):** Weights character n-grams by Zipf's Law, where order $k$ has weight $w_k \propto \frac{1}{k \cdot \ln(k+1)}$. This balances 1-gram tolerance (typos) against higher-order structural constraints.

- **Relative Order Consistency (ROC):** Measures if common characters maintain their relative sequence. Using the Longest Increasing Subsequence (LIS) of position indices, it penalizes semantic disruptions like word reordering ("我喜欢写代码" vs "代码写喜欢我") while tolerating local edits.

- **Position Dispersion ($\delta$):** Calculates the standard deviation of matching n-gram positions. Localized edits produce clustered matches (low $\delta$), while structural reorganizations scatter matches (high $\delta$).

The final score combines these with dynamic weights derived from text complexity $C$ and length ratio:

$$
\mathcal{F}(x, y) =
\left( \frac{L_{\min}}{L_{\max}} \right)^{\frac{\ln(L_{\min}+1)}{\ln(L_{\max}+1)}}
\times
\mathrm{ROC}^{\frac{1}{C+\varepsilon}}
\times
\frac{ S_{base} + (1 - |\delta_x - \delta_y|) \cdot w_{disp} }{ 1 + w_{disp} }
$$

### 2. Sequence Alignment: Needleman-Wunsch with Affine Gap

At the sequence level, the algorithm performs global alignment using dynamic programming. Unlike linear gap penalties, affine gaps distinguish between **opening** a gap ($g_{open}$) and **extending** it ($g_{extend}$). This prevents fragmentation—continuous deletions/insertions are treated as single structural edits rather than multiple independent operations.

A similarity threshold $\tau$ is applied: matches below this threshold are penalized more heavily than opening a gap, forcing low-similarity segments into separate "delete/insert" blocks rather than noisy "substitute" alignments.

### 3. Character-Level Refinement & Block Merging

After sentence alignment, character-level alignment is applied **only within matched segment pairs**. This constrains the $O(n^2)$ DP space.

A key innovation is **Difference Block Merging**: consecutive mismatched characters are aggregated into contiguous "delete" and "insert" chunks rather than interleaved operations. For example, transforming "ABC" to "XYZ" renders as `[-ABC][+XYZ]` instead of `[-A][+X][-B][+Y][-C][+Z]`, significantly improving visual clarity for end-users.

## Implementation

```python
from similarity_plus import hybrid_similarity
from alignment import NeedlemanWunschAligner
from char_aligner import get_rich_diff

# 1. Similarity scoring (no external libs required)
score = hybrid_similarity("今天天气很好", "今日气候宜人")
# Uses dynamic N-gram weights, ROC, and LCS internally

# 2. Sentence-level global alignment
aligner = NeedlemanWunschAligner(gap_open=-1.5, gap_extend=-0.2)
raw_alignment = aligner.align(sentences_a, sentences_b, similarity_func)

# 3. Character-level refinement with block merging
detailed_diff = get_rich_diff(raw_alignment)
```

## Design Constraints

The algorithm is designed for "extreme" scenarios where standard NLP tools fail:
- **No GCC/Compilers**: Pure Python implementation; uses bit operations for $\ln$ approximation.
- **Low Memory**: LCS uses $O(\min(n,m))$ space; character DP is scope-limited.
- **No Thresholds**: Weights are derived dynamically from text length and complexity, avoiding hard-coded constants.
