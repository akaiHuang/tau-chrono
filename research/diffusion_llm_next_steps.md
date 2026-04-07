# Diffusion LLM + tau: 研究方向分析（2026-03-20）

## 結論：noise schedule 是錯的方向

### 為什麼 schedule 沒效果
1. MDLM 的 ELBO 在連續極限下是 **schedule-invariant**（換 schedule 不影響最終收斂）
2. MDLM 的 masking 是 **獨立 per-token**，不是 channel composition → Petz 的核心假設不成立
3. β=1.0 和 β=0.3 結果一樣 = 確認了 schedule 不重要

### 正確的方向：推理時間（inference），不是訓練時間（training）

## 新發現：「先 unmask 簡單的 token」已經有人在做

| 方法 | 日期 | 用什麼信號 | 加速 |
|---|---|---|---|
| KLASS | 2025/11 | KL divergence | 2.78x + 品質提升 |
| CCD | 2025/11 | mutual information | 3.48x + 3.91% 品質提升 |
| Swordsman | 2026/02 | entropy | training-free |
| RL policy | 2025/12 | learned | outperforms heuristics |

**但沒有人用 Petz recovery theory 來解釋為什麼這些方法 work。**

## 我們的獨特貢獻空間

不是做新算法（別人已經做了），而是寫 **理論 paper**：

1. 證明：masked diffusion inference = Petz recovery map inversion
2. 證明：τ = 1 - max_v p(v) 是 recovery failure 的下界
3. 推導：最優 unmasking 順序可以從 Petz 框架得出
4. 證明：KLASS 和 CCD 是 Petz 框架的特例
5. 預測：現有方法在什麼情況下會失敗（Petz bound gap）

## 行動計畫

### 短期（不需要實驗）
- 寫理論 paper 連接 Petz recovery → diffusion inference
- 這是 Paper 1b 的自然延伸

### 中期（需要實驗）
- 實作 tau-guided unmasking，跟 KLASS/CCD 比較
- 找到 Petz bound 不 tight 的情況，證明我們的方法更好

### 長期
- tau 作為 RL 的 intrinsic reward（讓 diffusion LLM 生成更 coherent 的文字）
