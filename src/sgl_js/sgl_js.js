/**
 * sgl_js
 */

// 工具函数
const safeDiv = (a, b) => b !== 0 ? a / b : 0;
const getTextComplexity = s => s.length > 0 ? new Set(s).size / s.length : 0;

// sgltext - 文本预处理
const SENT_PATTERN = /([\u3002\uff01\uff1f.!?\u2026]+[\u201c\u201d\u2018\u2019\"\'\u300a\u300b\u3008\u3009\u300c\u300d]*|\n+)/g;

function splitSents(text, minlen = 1, filterpunct = true) {
    if (!text || !text.trim()) return [];
    text = text.trim();
    const parts = text.split(SENT_PATTERN);
    const sents = [];
    for (let i = 0; i < parts.length - 1; i += 2) {
        const c = (parts[i] + (parts[i + 1] || '')).trim();
        if (c) sents.push(c);
    }
    if (parts.length % 2 === 1 && parts[parts.length - 1].trim()) {
        sents.push(parts[parts.length - 1].trim());
    }
    const result = [];
    for (const s of sents) {
        const cleaned = s.replace(/\s+/g, ' ').trim();
        if (cleaned.length <= minlen) continue;
        if (filterpunct && !/[\w\u4e00-\u9fff]/.test(cleaned)) continue;
        result.push(cleaned);
    }
    return result.length > 0 ? result : [text];
}

const countSents = text => splitSents(text).length;

// sglsim - 混合相似度计算
const similarityCache = new Map();
const CACHE_KEY_PREFIX = 'sgl_sim_';
const getCacheKey = (x, y) => `${CACHE_KEY_PREFIX}${x.length}_${y.length}_${x}_${y}`;

function diceCoefficientWithPos(x, y, n) {
    const gx = [], gy = [];
    for (let i = 0; i <= x.length - n; i++) gx.push(x.substring(i, i + n));
    for (let i = 0; i <= y.length - n; i++) gy.push(y.substring(i, i + n));
    const cx = {}, cy = {};
    gx.forEach((g, i) => { if (!cx[g]) cx[g] = []; cx[g].push(i); });
    gy.forEach((g, i) => { if (!cy[g]) cy[g] = []; cy[g].push(i); });
    let overlap = 0;
    const px = [], py = [];
    for (const g in cx) {
        if (cy[g]) {
            const c = Math.min(cx[g].length, cy[g].length);
            overlap += c;
            px.push(...cx[g].slice(0, c));
            py.push(...cy[g].slice(0, c));
        }
    }
    return [safeDiv(2.0 * overlap, gx.length + gy.length), px, py];
}

const getLisLength = arr => {
    const tails = [];
    for (const num of arr) {
        let lo = 0, hi = tails.length;
        while (lo < hi) {
            const mid = Math.floor((lo + hi) / 2);
            if (tails[mid] < num) lo = mid + 1;
            else hi = mid;
        }
        if (lo === tails.length) tails.push(num);
        else tails[lo] = num;
    }
    return tails.length;
};

function computeRoc(x, y) {
    const yPos = {};
    for (let i = 0; i < y.length; i++) {
        if (!yPos[y[i]]) yPos[y[i]] = [];
        yPos[y[i]].push(i);
    }
    const indices = [];
    const ptr = {};
    let common = 0;
    for (const ch of x) {
        if (yPos[ch] && (ptr[ch] || 0) < yPos[ch].length) {
            indices.push(yPos[ch][ptr[ch] || 0]);
            ptr[ch] = (ptr[ch] || 0) + 1;
            common++;
        }
    }
    return common === 0 ? 0.0 : getLisLength(indices) / common;
}

function lcsRatio(x, y) {
    const n = x.length, m = y.length;
    if (n * m === 0) return 0.0;
    let prev = new Array(m + 1).fill(0);
    for (let i = 1; i <= n; i++) {
        const cur = new Array(m + 1).fill(0);
        const xi = x[i - 1];
        for (let j = 1; j <= m; j++) {
            cur[j] = xi === y[j - 1] ? prev[j - 1] + 1 : Math.max(prev[j], cur[j - 1]);
        }
        prev = cur;
    }
    return (2.0 * prev[m]) / (n + m);
}

const computeDispersion = (positions, textLength) => {
    if (positions.length <= 1) return 0.5;
    const mean = positions.reduce((a, b) => a + b, 0) / positions.length;
    const variance = positions.reduce((sum, pos) => sum + Math.pow(pos - mean, 2), 0) / positions.length;
    const std = Math.sqrt(variance);
    return Math.min(std / (textLength / 2.0 + 1e-9), 1.0);
};

const computeWeights = (textLength, nGramOrder) => {
    if (nGramOrder <= 1) return [1.0];
    const w = [];
    for (let k = 1; k <= nGramOrder; k++) {
        w.push(1.0 / (k * Math.log(k + 1)));
    }
    const t = w.reduce((a, b) => a + b, 0);
    return t > 0 ? w.map(v => v / t) : new Array(nGramOrder).fill(1.0 / nGramOrder);
};

function computeFull(x, y) {
    if (!x || !y) {
        return { final: 0.0, diceSum: 0.0, lcsScore: 0.0, rocScore: 0.0, complexityX: 0.0, complexityY: 0.0, nGramOrder: 1, structRatio: 0.0, logicReward: 0.0, lenPenalty: 0.0, dispersionConsistency: 0.0, confidence: 0.0 };
    }
    const lenMin = Math.min(x.length, y.length);
    const lenMax = Math.max(x.length, y.length);
    const alphabet = new Set(x + y).size;
    const complexity = (getTextComplexity(x) + getTextComplexity(y)) / 2.0;
    const nGramOrder = Math.max(1, lenMin.toString(2).length - 1);
    const weights = computeWeights(lenMin, nGramOrder);
    let diceSum = 0.0;
    const allPx = [], allPy = [];
    let matchCount = 0;
    const logLen = (Math.log(x.length + 1) + Math.log(y.length + 1)) / 2.0 + 1.0;
    for (let i = 0; i < nGramOrder; i++) {
        const order = i + 1;
        const [ds, px, py] = diceCoefficientWithPos(x, y, order);
        diceSum += weights[i] * ds;
        if (order === 1) matchCount = px.length;
        if (order <= logLen) { allPx.push(...px); allPy.push(...py); }
    }
    const lcsScore = lcsRatio(x, y);
    const rocScore = computeRoc(x, y);
    const lnLenMin = Math.log(lenMin + 1);
    const lnContext = Math.log(lenMax + alphabet + 1);
    const structRatio = safeDiv(lnLenMin, lnContext) * complexity;
    const baseScore = diceSum * (1.0 - structRatio) + lcsScore * structRatio;
    let dispersionConsistency = 0.0;
    if (allPx.length > 0 && allPy.length > 0) {
        const dx = computeDispersion(allPx, x.length);
        const dy = computeDispersion(allPy, y.length);
        dispersionConsistency = 1.0 - Math.abs(dx - dy);
    }
    const expectedRandom = 1.0 / (alphabet + 1.0);
    const confidence = safeDiv(matchCount / lenMin - expectedRandom, 1.0 - expectedRandom);
    const dispWeight = Math.max(0.0, confidence) * complexity;
    const logicReward = Math.pow(rocScore, 1.0 / (complexity + 1e-9));
    const gamma = safeDiv(lnLenMin, Math.log(lenMax + 1));
    const lenPenalty = Math.pow(lenMin / lenMax, gamma);
    const final = lenPenalty * logicReward * (baseScore + dispersionConsistency * dispWeight) / (1.0 + dispWeight);
    return { final, diceSum, lcsScore, rocScore, complexityX: getTextComplexity(x), complexityY: getTextComplexity(y), nGramOrder, structRatio, logicReward, lenPenalty, dispersionConsistency, confidence };
}

function hybridSimilarity(x, y) {
    const key = getCacheKey(x, y);
    if (similarityCache.has(key)) {
        similarityCache._hits = (similarityCache._hits || 0) + 1;
        return similarityCache.get(key);
    }
    similarityCache._misses = (similarityCache._misses || 0) + 1;
    const result = computeFull(x, y);
    const score = parseFloat(result.final.toFixed(4));
    similarityCache.set(key, score);
    return score;
}

function hybridDetail(x, y) {
    const r = computeFull(x, y);
    const order = r.nGramOrder;
    const w = computeWeights(Math.min(x.length, y.length), order);
    const obj = {
        score: parseFloat(r.final.toFixed(4)),
        diceScore: parseFloat(r.diceSum.toFixed(4)),
        lcsScore: parseFloat(r.lcsScore.toFixed(4)),
        rocScore: parseFloat(r.rocScore.toFixed(4)),
        complexityA: parseFloat(r.complexityX.toFixed(4)),
        complexityB: parseFloat(r.complexityY.toFixed(4)),
        nGramOrder: order,
        nGramWeights: Object.fromEntries(w.map((v, i) => [i + 1, parseFloat(v.toFixed(4))])),
        logicReward: parseFloat(r.logicReward.toFixed(4)),
        lenPenalty: parseFloat(r.lenPenalty.toFixed(4)),
        dispersionConsistency: parseFloat(r.dispersionConsistency.toFixed(4)),
        structRatio: parseFloat(r.structRatio.toFixed(4)),
        confidence: parseFloat(r.confidence.toFixed(4))
    };
    obj.toDict = () => ({ score: obj.score, dice_score: obj.diceScore, lcs_score: obj.lcsScore, roc_score: obj.rocScore, complexity_a: obj.complexityA, complexity_b: obj.complexityB, n_gram_order: obj.nGramOrder, n_gram_weights: obj.nGramWeights, logic_reward: obj.logicReward, len_penalty: obj.lenPenalty, dispersion_consistency: obj.dispersionConsistency, struct_ratio: obj.structRatio, confidence: obj.confidence });
    obj.toJson = (indent = 2) => JSON.stringify(obj.toDict(), null, indent);
    obj.summary = () => ({ score: obj.score, dice: obj.diceScore, lcs: obj.lcsScore, roc: obj.rocScore, reward: obj.logicReward });
    return obj;
}

const cacheClear = () => similarityCache.clear();
const cacheInfo = () => ({ currsize: similarityCache.size, hits: similarityCache._hits || 0, misses: similarityCache._misses || 0 });

// sglalign - Needleman-Wunsch 全局对齐
class Aligner {
    constructor(gapOpen = -1.5, gapExtend = -0.2, mismatchThreshold = 0.2) {
        this.gapOpen = gapOpen;
        this.gapExtend = gapExtend;
        this.mismatchThreshold = mismatchThreshold;
    }
    align(seq1, seq2, similarityFunc) {
        const n = seq1.length, m = seq2.length;
        const gO = this.gapOpen, gE = this.gapExtend, th = this.mismatchThreshold;
        const M = Array(n + 1).fill(null).map(() => Array(m + 1).fill(-Infinity));
        const X = Array(n + 1).fill(null).map(() => Array(m + 1).fill(-Infinity));
        const Y = Array(n + 1).fill(null).map(() => Array(m + 1).fill(-Infinity));
        M[0][0] = 0.0;
        for (let i = 1; i <= n; i++) X[i][0] = gO + (i - 1) * gE;
        for (let j = 1; j <= m; j++) Y[0][j] = gO + (j - 1) * gE;
        for (let i = 1; i <= n; i++) {
            const si = seq1[i - 1];
            const simRow = seq2.map(sj => similarityFunc(si, sj));
            const scores = simRow.map(s => s >= th ? s : gO * 1.5);
            for (let j = 1; j <= m; j++) {
                const diagMax = Math.max(M[i - 1][j - 1], X[i - 1][j - 1], Y[i - 1][j - 1]);
                M[i][j] = diagMax + scores[j - 1];
            }
            for (let j = 1; j <= m; j++) {
                X[i][j] = Math.max(M[i - 1][j] + gO, X[i - 1][j] + gE);
            }
            const mPlusOpen = M[i].slice(0, m).map(v => v + gO);
            const yRow = [Y[i][0]];
            for (let j = 1; j <= m; j++) {
                const cand = mPlusOpen[j - 1];
                yRow.push(Math.max(cand, yRow[j - 1] + gE));
            }
            Y[i] = yRow;
        }
        const combined = Array(n + 1).fill(null).map((_, i) => Array(m + 1).fill(null).map((_, j) => Math.max(M[i][j], X[i][j], Y[i][j])));
        const totalScore = combined[n][m];
        const [pairs, path] = this._backtrace(seq1, seq2, M, X, Y, similarityFunc);
        const result = {
            pairs, score: totalScore, nSource: n, nTarget: m, sourceSeqs: [...seq1], targetSeqs: [...seq2],
            dpMatrix: combined, matrixM: M, matrixX: X, matrixY: Y, backtracePath: path,
            fbMatrix: null, shuffleGroups: null, gapOpen: gO, gapExtend: gE, mismatchThreshold: th
        };
        result.toDict = () => ({ pairs: result.pairs.map(p => ({ source: p.source, target: p.target, similarity: p.similarity, isGap: p.isGap, state: p.state, isShuffled: p.isShuffled })), score: result.score, n_source: result.nSource, n_target: result.nTarget, source_seqs: result.sourceSeqs, target_seqs: result.targetSeqs, dp_matrix: result.dpMatrix, matrix_m: result.matrixM, matrix_x: result.matrixX, matrix_y: result.matrixY, backtrace_path: result.backtracePath, gap_open: result.gapOpen, gap_extend: result.gapExtend, mismatch_threshold: result.mismatchThreshold });
        result.toJson = (indent = 2) => JSON.stringify(result.toDict(), null, indent);
        result.summary = () => ({ score: parseFloat(result.score.toFixed(4)), n_pairs: result.pairs.length, matches: result.pairs.filter(p => p.state === 'match').length });
        return result;
    }
    alignfb(seq1, seq2, similarityFunc) {
        const result = this.align(seq1, seq2, similarityFunc);
        result.fbMatrix = this._computeFb(seq1, seq2, similarityFunc);
        return result;
    }
    reorderAlign(seq1, seq2, similarityFunc, threshold = 0.3) {
        const n = seq1.length, m = seq2.length;
        const simMat = this._computeSimMatrix(seq1, seq2, similarityFunc);
        const matches = this._greedyMatch(simMat, threshold);
        matches.sort((a, b) => a[0] - b[0]);
        const tgtSeq = matches.map(p => p[1]);
        const shuffleGroups = this._detectShuffleGroups(matches, tgtSeq);
        const shuffledSrc = new Set(shuffleGroups.flatMap(g => g.sourceIndices));
        const matchMap = new Map(matches.map(p => [p[0], [p[1], p[2]]]));
        const matchedTgt = new Set(matches.map(p => p[1]));
        const pairs = [];
        let si = 0, tj = 0;
        while (si < n || tj < m) {
            if (si < n && matchMap.has(si)) {
                const [j, sim] = matchMap.get(si);
                while (tj < j) {
                    if (!matchedTgt.has(tj)) {
                        pairs.push({ source: null, target: seq2[tj], similarity: 0.0, isGap: true, state: 'insert', isShuffled: false });
                    }
                    tj++;
                }
                pairs.push({ source: seq1[si], target: seq2[j], similarity: sim, isGap: false, state: sim >= this.mismatchThreshold ? 'match' : 'mismatch', isShuffled: shuffledSrc.has(si) });
                si++;
                tj = Math.max(tj, j + 1);
            } else if (si < n) {
                pairs.push({ source: seq1[si], target: null, similarity: 0.0, isGap: true, state: 'delete', isShuffled: false });
                si++;
            } else {
                if (!matchedTgt.has(tj)) {
                    pairs.push({ source: null, target: seq2[tj], similarity: 0.0, isGap: true, state: 'insert', isShuffled: false });
                }
                tj++;
            }
        }
        const totalScore = pairs.filter(p => !p.isGap).reduce((sum, p) => sum + p.similarity, 0);
        const result = { pairs, score: totalScore, nSource: n, nTarget: m, sourceSeqs: [...seq1], targetSeqs: [...seq2], dpMatrix: null, matrixM: null, matrixX: null, matrixY: null, backtracePath: null, fbMatrix: null, shuffleGroups, gapOpen: this.gapOpen, gapExtend: this.gapExtend, mismatchThreshold: this.mismatchThreshold };
        result.toDict = () => ({ pairs: result.pairs.map(p => ({ source: p.source, target: p.target, similarity: p.similarity, isGap: p.isGap, state: p.state, isShuffled: p.isShuffled })), score: result.score, n_source: result.nSource, n_target: result.nTarget, source_seqs: result.sourceSeqs, target_seqs: result.targetSeqs, shuffle_groups: result.shuffleGroups.map(g => ({ source_indices: g.sourceIndices, target_indices: g.targetIndices })), gap_open: result.gapOpen, gap_extend: result.gapExtend, mismatch_threshold: result.mismatchThreshold });
        result.toJson = (indent = 2) => JSON.stringify(result.toDict(), null, indent);
        result.summary = () => ({ score: parseFloat(result.score.toFixed(4)), n_pairs: result.pairs.length, shuffle_groups: result.shuffleGroups.length });
        return result;
    }
    _computeSimMatrix(seq1, seq2, simFunc) {
        const n = seq1.length, m = seq2.length;
        const mat = Array(n).fill(null).map(() => Array(m).fill(0));
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < m; j++) {
                mat[i][j] = simFunc(seq1[i], seq2[j]);
            }
        }
        return mat;
    }
    _greedyMatch(simMat, threshold) {
        const n = simMat.length, m = simMat[0].length;
        const flat = [];
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < m; j++) {
                if (simMat[i][j] >= threshold) {
                    flat.push([i, j, simMat[i][j]]);
                }
            }
        }
        flat.sort((a, b) => b[2] - a[2]);
        const matches = [];
        const usedSrc = new Set(), usedTgt = new Set();
        for (const [i, j, s] of flat) {
            if (!usedSrc.has(i) && !usedTgt.has(j)) {
                matches.push([i, j, s]);
                usedSrc.add(i);
                usedTgt.add(j);
            }
        }
        return matches;
    }
    _detectShuffleGroups(matches, tgtSeq) {
        if (matches.length < 2) return [];
        const expected = [...tgtSeq].sort((a, b) => a - b);
        const bad = [];
        for (let k = 0; k < matches.length; k++) {
            if (tgtSeq[k] !== expected[k]) {
                bad.push(k);
            }
        }
        if (bad.length < 2) return [];
        const groups = [[bad[0]]];
        for (let i = 1; i < bad.length; i++) {
            if (bad[i] === groups[groups.length - 1][groups[groups.length - 1].length - 1] + 1) {
                groups[groups.length - 1].push(bad[i]);
            } else {
                groups.push([bad[i]]);
            }
        }
        return groups.filter(g => g.length >= 2).map(g => ({ sourceIndices: g.map(k => matches[k][0]), targetIndices: g.map(k => matches[k][1]) }));
    }
    _backtrace(seq1, seq2, M, X, Y, simFunc) {
        const n = seq1.length, m = seq2.length;
        const pairs = [], path = [];
        let i = n, j = m;
        while (i > 0 || j > 0) {
            if (i === 0) {
                pairs.unshift({ source: null, target: seq2[j - 1], similarity: 0.0, isGap: true, state: 'insert', isShuffled: false });
                path.unshift([i, j, 'Y']);
                j--;
            } else if (j === 0) {
                pairs.unshift({ source: seq1[i - 1], target: null, similarity: 0.0, isGap: true, state: 'delete', isShuffled: false });
                path.unshift([i, j, 'X']);
                i--;
            } else {
                const combined = Math.max(M[i][j], X[i][j], Y[i][j]);
                if (Math.abs(M[i][j] - combined) < 1e-10) {
                    const sim = simFunc(seq1[i - 1], seq2[j - 1]);
                    pairs.unshift({ source: seq1[i - 1], target: seq2[j - 1], similarity: sim, isGap: false, state: sim >= this.mismatchThreshold ? 'match' : 'mismatch', isShuffled: false });
                    path.unshift([i, j, 'M']);
                    i--;
                    j--;
                } else if (Math.abs(X[i][j] - combined) < 1e-10) {
                    pairs.unshift({ source: seq1[i - 1], target: null, similarity: 0.0, isGap: true, state: 'delete', isShuffled: false });
                    path.unshift([i, j, 'X']);
                    i--;
                } else {
                    pairs.unshift({ source: null, target: seq2[j - 1], similarity: 0.0, isGap: true, state: 'insert', isShuffled: false });
                    path.unshift([i, j, 'Y']);
                    j--;
                }
            }
        }
        path.unshift([0, 0, 'M']);
        return [pairs, path];
    }
    _computeFb(seq1, seq2, simFunc) {
        const n = seq1.length, m = seq2.length;
        const fb = Array(n + 1).fill(null).map(() => Array(m + 1).fill(0.0));
        for (let i = 1; i <= n; i++) {
            for (let j = 1; j <= m; j++) {
                fb[i][j] = fb[i - 1][j - 1] + simFunc(seq1[i - 1], seq2[j - 1]);
            }
        }
        return fb;
    }
}

// sgldiff - 字符级差异对齐
const _charSim = (c1, c2) => c1 === c2 ? 1.0 : 0.0;

class CharLevelAligner {
    constructor() {
        this.aligner = new Aligner(-0.8, -0.8, 0.1);
    }
    alignChars(text1, text2) {
        if (!text1) {
            return text2.split('').map(c => ({ source: null, target: c, diffType: 'insert' }));
        }
        if (!text2) {
            return text1.split('').map(c => ({ source: c, target: null, diffType: 'delete' }));
        }
        const raw = this.aligner.align(text1.split(''), text2.split(''), _charSim).pairs;
        const refined = [];
        let i = 0, n = raw.length;
        while (i < n) {
            const p = raw[i];
            if (p.source && p.target && p.source === p.target) {
                refined.push({ source: p.source, target: p.target, diffType: 'equal' });
                i++;
                continue;
            }
            const orig = [], rew = [];
            while (i < n) {
                const cp = raw[i];
                if (cp.source && cp.target && cp.source === cp.target) break;
                if (cp.source) orig.push(cp.source);
                if (cp.target) rew.push(cp.target);
                i++;
            }
            for (const c of orig) {
                refined.push({ source: c, target: null, diffType: 'delete' });
            }
            for (const c of rew) {
                refined.push({ source: null, target: c, diffType: 'insert' });
            }
        }
        return refined;
    }
}

const ANCHOR = 0.80;

function richDiff(alignmentResult, similarityFunc = null, anchorThreshold = ANCHOR) {
    const ca = new CharLevelAligner();
    const anchor = (s1, s2) => !s1 || !s2 ? false : s1 === s2 || (similarityFunc !== null && similarityFunc(s1, s2) >= anchorThreshold);
    const blocks = [];
    let i = 0, n = alignmentResult.length;
    while (i < n) {
        const p = alignmentResult[i];
        if (anchor(p.source, p.target)) {
            if (p.source === p.target) {
                blocks.push({ blockType: 'equal', source: p.source, target: p.target, charDiffs: p.source.split('').map(c => ({ source: c, target: c, diffType: 'equal' })) });
            } else {
                blocks.push({ blockType: 'modify', source: p.source, target: p.target, charDiffs: ca.alignChars(p.source, p.target) });
            }
            i++;
            continue;
        }
        const orig = [], rew = [];
        while (i < n) {
            const cp = alignmentResult[i];
            if (anchor(cp.source, cp.target)) break;
            if (cp.source) orig.push(cp.source);
            if (cp.target) rew.push(cp.target);
            i++;
        }
        const fo = orig.join(''), fr = rew.join('');
        if (fo && fr) {
            blocks.push({ blockType: 'modify', source: fo, target: fr, charDiffs: ca.alignChars(fo, fr) });
        } else if (fo) {
            blocks.push({ blockType: 'delete', source: fo, target: null, charDiffs: fo.split('').map(c => ({ source: c, target: null, diffType: 'delete' })) });
        } else if (fr) {
            blocks.push({ blockType: 'insert', source: null, target: fr, charDiffs: fr.split('').map(c => ({ source: null, target: c, diffType: 'insert' })) });
        }
    }
    const result = {
        blocks,
        summary() {
            const summary = {};
            for (const b of this.blocks) {
                summary[b.blockType] = (summary[b.blockType] || 0) + 1;
            }
            return summary;
        },
        toDiffText() {
            const lines = [];
            for (const b of this.blocks) {
                if (b.blockType === 'equal') {
                    lines.push(`  ${b.source}`);
                } else if (b.blockType === 'delete') {
                    lines.push(`- ${b.source}`);
                } else if (b.blockType === 'insert') {
                    lines.push(`+ ${b.target}`);
                } else if (b.blockType === 'modify') {
                    lines.push(`- ${b.source}`);
                    lines.push(`+ ${b.target}`);
                }
            }
            return lines.join('\n');
        }
    };
    return result;
}

const charDiff = (text1, text2) => new CharLevelAligner().alignChars(text1, text2);

// 导出
const sgl = {
    hybridSimilarity, hybridDetail, cacheClear, cacheInfo,
    Aligner,
    richDiff, charDiff,
    splitSents, countSents
};

if (typeof module !== 'undefined' && module.exports) {
    module.exports = sgl;
}

if (typeof window !== 'undefined') {
    window.sgl = sgl;
}
