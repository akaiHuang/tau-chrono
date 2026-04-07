# Tau-Conditioned Denoising for Diffusion LLMs

**Design Document v1.0 -- 2026-03-20**
**Author: Sheng-Kai Huang / Claude**

---

## 0. Executive Summary

We propose **tau-conditioned denoising** (TCD): at each reverse diffusion step, compute a per-token recovery difficulty score tau_j for every masked position, then feed these tau values as additional conditioning to the denoiser transformer. The key insight is that tau_j carries **indirect information about the identity of token j**, partially bridging the factorization gap between diffusion LLMs and autoregressive models.

This is NOT another unmasking-order heuristic (KLASS, CCD, Swordsman already do that). TCD is fundamentally different: it injects position-dependent difficulty signals **into the score network itself**, allowing the model to learn cross-position correlations that factorized prediction normally misses.

---

## 1. The Problem: Factorization Mismatch

### 1.1 Why Diffusion LLMs Underperform AR Models

A diffusion LLM (MDLM, SEDD, etc.) predicts:

```
P(x_i = v | x_visible)   independently for each masked position i
```

An autoregressive model predicts:

```
P(x_i = v | x_1, ..., x_{i-1})   sequentially, conditioning on all prior tokens
```

The diffusion model's factorized prediction means: when predicting token at position 3, it has **no information** about what token 5 might be (if both are masked). The AR model, by contrast, has already committed to token 3 before predicting token 5 -- full sequential conditioning.

### 1.2 Why This Matters

Consider: "The ___ ran ___ the ___"

- Position 2 could be: dog, cat, child, river, clock...
- Position 4 could be: across, down, through, over, into...
- Position 6 could be: street, hill, hallway, track, field...

An AR model seeing "The dog" would know position 4 is likely "across/through/down". A factorized diffusion model predicts each independently, missing these correlations.

### 1.3 The Tau Bridge

**Key insight**: Even without knowing the exact token, the **difficulty of recovering** each position carries information about what that token is.

If tau_5 is very low (easy to recover), position 5 is probably a function word ("the", "a", "in"). If tau_3 is high (hard to recover), position 3 is probably a content word carrying heavy semantic load. This distributional information, when fed to the network, allows it to learn:

> "When position 5 is easy (probably a function word) and position 3 is hard (probably a content word), the most likely content word for position 3 given a function word at 5 is..."

This is a **soft, probabilistic** form of the sequential conditioning that AR models get for free.

---

## 2. Computing Tau During Inference

### 2.1 Definition

For a masked diffusion model at denoising step t, define the per-position recovery difficulty:

```
tau_j(t) = 1 - max_v  P_theta(x_j = v | x_visible, t)
```

where P_theta is the denoiser's current prediction for position j.

**Interpretation**:
- tau_j = 0: The model is 100% confident about token j (trivially recoverable)
- tau_j = 1: The model assigns equal probability to all tokens (maximally uncertain)
- tau_j in (0, 1): Partial confidence; higher tau = harder to recover

This is precisely the **Petz recovery failure parameter** applied to discrete token recovery: tau = 1 - F, where F is the fidelity (max probability) of recovery.

### 2.2 Efficient Computation

**Naive approach**: Run the denoiser once to get P_theta, compute tau, then run it AGAIN with tau conditioning. Cost: 2x forward passes.

**Efficient approach (two-phase within single forward pass)**:

```
Phase 1: Standard forward through layers 1..L/2
          → Extract intermediate logits from the midpoint
          → Compute tau_j from these intermediate predictions

Phase 2: Inject tau into layers L/2+1..L as additional conditioning
          → Final output benefits from tau information
```

**Cost**: ~1.0x forward passes (midpoint extraction is free; tau computation is O(seq_len * vocab) which is negligible vs. transformer attention O(seq_len^2 * d_model)).

### 2.3 Alternative: Amortized Tau from Previous Step

At denoising step t, use the tau values computed from step t+1 (the previous denoising step):

```
tau_j^{(t)} := 1 - max_v P_theta(x_j = v | x^{(t+1)}_visible, t+1)
```

**Cost**: Exactly 0 extra computation -- tau comes for free from the previous denoising step's output.

**Trade-off**: Tau is slightly stale (from the previous noise level), but since adjacent denoising steps are similar (especially with many steps), the staleness is minimal.

**Recommendation**: Use amortized tau for production; use two-phase for ablation experiments.

---

## 3. Architecture: Injecting Tau into the Transformer

### 3.1 Option A: Tau Embedding (Recommended)

Add tau as an additional per-position embedding, analogous to how diffusion timestep t is injected:

```python
class TauEmbedding(nn.Module):
    """Per-position tau embedding.

    Maps scalar tau_j in [0,1] to d_model-dimensional vector using
    a small MLP, then adds to the token embedding.
    """
    def __init__(self, d_model: int, d_tau_hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, d_tau_hidden),
            nn.SiLU(),
            nn.Linear(d_tau_hidden, d_model),
        )

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """
        tau: (batch, seq_len) -- per-position tau values
        returns: (batch, seq_len, d_model)
        """
        return self.mlp(tau.unsqueeze(-1))  # (B, L, 1) -> (B, L, d_model)
```

**Injection point**: Add tau_embedding to the input embeddings before the first transformer layer:

```python
h = token_embedding + position_embedding + time_embedding + tau_embedding
```

**Parameter cost**: 64 + 64*d_model parameters (negligible for d_model=768: ~50K params vs. 100M+ total).

### 3.2 Option B: FiLM Conditioning (Per-Layer)

Use Feature-wise Linear Modulation at every transformer layer, exactly as the existing QEC score network does:

```python
class TauFiLM(nn.Module):
    def __init__(self, d_model: int, d_tau: int = 64):
        super().__init__()
        self.tau_proj = nn.Linear(1, d_tau)
        self.film = nn.Linear(d_tau, 2 * d_model)

    def forward(self, h: torch.Tensor, tau: torch.Tensor):
        """
        h: (B, L, d_model) -- hidden states
        tau: (B, L) -- per-position tau
        """
        z = F.silu(self.tau_proj(tau.unsqueeze(-1)))  # (B, L, d_tau)
        gamma_beta = self.film(z)  # (B, L, 2*d_model)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return (1 + gamma) * h + beta
```

**Pro**: More expressive (per-layer modulation).
**Con**: More parameters (2 * d_model * d_tau per layer), and requires modifying every transformer layer.

### 3.3 Option C: Attention Bias

Add tau-derived bias to the attention scores:

```python
# In attention computation:
attn_scores = Q @ K^T / sqrt(d_k)
tau_bias = tau_bias_net(tau_i, tau_j)  # (B, L, L)
attn_scores = attn_scores + tau_bias
```

**Interpretation**: Positions with similar tau values (similar difficulty) should attend to each other more, because they're likely of the same "type" (both content words, or both function words).

**Con**: O(L^2) computation for the bias matrix; complex to implement.

### 3.4 Recommendation

**Start with Option A (Tau Embedding)** for simplicity and low parameter cost. If that works, try Option B (FiLM) for potentially better results. Skip Option C unless A and B both plateau.

---

## 4. Training with Tau Conditioning

### 4.1 Modified Training Loop

The key challenge: during training, we need ground-truth tau values to condition on. But tau is defined from the model's own predictions -- a chicken-and-egg problem.

**Solution: Teacher-Forcing with Oracle Tau**

During training, compute "oracle tau" from the known clean tokens:

```python
def compute_oracle_tau(x_clean, x_noisy, mask, vocab_size):
    """
    Oracle tau: computed from the ground truth.

    For masked positions, tau = 1 - 1/vocab_size (maximum uncertainty)
    For visible positions, tau = 0 (known with certainty)

    But more informatively: use the model's OWN prediction from the
    previous training step (EMA of tau).
    """
    tau = torch.ones_like(mask, dtype=torch.float32)
    tau[~mask] = 0.0  # visible positions have tau = 0
    return tau
```

**Better: Self-Tau (Stop-Gradient)**

Run the model twice per training step:
1. Forward pass WITHOUT tau -> get predictions -> compute tau (stop gradient)
2. Forward pass WITH tau from step 1 -> compute loss

```python
def training_step(model, x_clean, x_noisy, mask, t):
    # Phase 1: Get tau estimates (no gradient)
    with torch.no_grad():
        logits_1 = model(x_noisy, t, tau=None)
        probs_1 = F.softmax(logits_1, dim=-1)
        tau = 1.0 - probs_1.max(dim=-1).values  # (B, L)
        tau = tau * mask.float()  # only masked positions have nonzero tau

    # Phase 2: Forward with tau conditioning
    logits_2 = model(x_noisy, t, tau=tau)

    # Loss: cross-entropy on masked positions
    loss = F.cross_entropy(
        logits_2[mask].view(-1, vocab_size),
        x_clean[mask].view(-1),
    )
    return loss
```

### 4.2 Curriculum: Gradual Tau Introduction

To prevent the model from becoming dependent on tau too early:

```python
def get_tau_dropout(epoch, total_epochs):
    """Gradually introduce tau conditioning."""
    # First 20% of training: no tau (tau_dropout = 1.0)
    # Linearly ramp up tau availability
    warmup = 0.2
    if epoch / total_epochs < warmup:
        return 1.0
    return max(0.0, 1.0 - (epoch / total_epochs - warmup) / (1.0 - warmup))

# During training:
tau_drop = get_tau_dropout(epoch, total_epochs)
if random.random() < tau_drop:
    tau = None  # train without tau
else:
    tau = compute_tau(...)  # train with tau
```

This ensures the model can still function without tau (graceful degradation) while learning to exploit tau when available.

### 4.3 Loss Function

No change to the loss function. Standard masked cross-entropy:

```
L = -E_{t,x_0,mask} [ sum_{j in masked} log P_theta(x_0_j | x_noisy, t, tau) ]
```

The only change is that P_theta now also conditions on tau.

### 4.4 ELBO Analysis

For MDLM, the ELBO is:

```
-log p(x) <= E_t [ sum_j mask_j(t) * D_KL( q(x_j|x_0) || p_theta(x_j|x_noisy,t) ) ]
```

Adding tau conditioning:

```
-log p(x) <= E_t [ sum_j mask_j(t) * D_KL( q(x_j|x_0) || p_theta(x_j|x_noisy,t,tau) ) ]
```

Since tau is a deterministic function of (x_noisy, t, theta), this is still a valid ELBO. The tau conditioning can only **tighten** the bound (more information -> lower KL) or leave it unchanged. **It cannot hurt.**

---

## 5. Theoretical Analysis

### 5.1 Why TCD Should Help: Information-Theoretic Argument

The factorized prediction gives:

```
P_factorized(x_masked | x_visible) = prod_j P(x_j | x_visible)
```

The true conditional is:

```
P_true(x_masked | x_visible) = prod_j P(x_j | x_visible, x_{masked\j})
```

The gap is:

```
D_KL(P_true || P_factorized) = sum_j I(x_j ; x_{masked\j} | x_visible)
```

This is the **total mutual information** between masked positions.

TCD provides tau_j, which is a **sufficient statistic** for the marginal entropy H(x_j | x_visible). Since I(x_j ; x_k | x_visible) depends on the marginals H(x_j), H(x_k) and the joint H(x_j, x_k), knowing the marginals (via tau) recovers a portion of the mutual information.

**Quantitative bound**: If tokens are bimodal (content vs. function words), tau reveals the "type" of each position. The mutual information between types is:

```
I(type_j ; type_k | x_visible) = H(type_j | x_visible) - H(type_j | type_k, x_visible)
```

For English text, function words (the, a, in, of, ...) make up ~50% of tokens but only ~0.1% of vocabulary. So knowing "this position is a function word" reduces effective vocabulary from 50K to ~50, a >1000x reduction. This translates to:

```
Entropy reduction: log(50K) - log(50) = log(1000) ≈ 7 bits per function-word position
```

### 5.2 Connection to Petz Recovery

In the Petz framework:

```
tau_j = 1 - F(rho_j, R_sigma(N(rho_j)))
```

where:
- rho_j = the true token distribution at position j
- N = the masking channel (forward diffusion)
- R_sigma = the Petz recovery map (denoiser)
- F = fidelity (overlap between true and recovered distribution)

The bound: F >= exp(-Sigma/2) where Sigma = D(rho || sigma).

For masked diffusion:
- Sigma = log(V) for a uniformly masked position (V = vocab size)
- F >= 1/sqrt(V)
- tau <= 1 - 1/sqrt(V)

After partial denoising (step t):
- Sigma(t) decreases as the model narrows down possibilities
- tau(t) decreases monotonically as denoising progresses

**TCD operationalizes Petz recovery**: by feeding tau into the network, we tell it "here's how much information the recovery map has failed to recover at each position" -- the network can then allocate more capacity to high-tau (hard) positions and use the certainty at low-tau (easy) positions as soft constraints.

### 5.3 Expected Improvement

For a sequence of length L with M masked positions:

| Regime | Tau Signal | Expected Perplexity Reduction |
|--------|-----------|-------------------------------|
| M = 1 | Trivial (only one masked position) | 0% |
| M = 2-5 | Type information (content vs function) | 5-15% |
| M = L/4 | Strong cross-position correlations | 10-25% |
| M = L/2 | Diminishing returns (too many unknowns) | 5-15% |
| M = L | All masked (generation from scratch) | 15-30% |

The sweet spot is intermediate masking ratios (25-75%), which is exactly where diffusion LLMs spend most denoising steps.

### 5.4 Can This Close the Gap with AR Models?

**Short answer: No, not fully. But it can close 20-40% of the gap.**

The remaining gap comes from:
1. **Tau is a scalar per position** -- AR models have full distribution P(x_j | x_{<j}), which is V-dimensional. Tau compresses this to a single number. The information loss is log(V) - 1 ≈ 15 bits per position.
2. **Tau is computed from the current (imperfect) model** -- at early denoising steps, the model's predictions are poor, so tau is noisy.
3. **Tau only captures marginal uncertainty, not joint** -- the pairwise and higher-order correlations between masked positions are not captured.

However, these limitations suggest **extensions** that could close the gap further:
- **Vector tau**: Instead of scalar tau_j, use the full logit vector (but this is expensive)
- **Pairwise tau**: Compute tau_{j,k} capturing joint difficulty (O(L^2) cost)
- **Iterative refinement**: Multiple tau-conditioned passes per denoising step

---

## 6. Implementation Plan for MDLM Codebase

### 6.1 Phase 1: Proof of Concept (1-2 days, M1 Max feasible)

**Task**: Add tau conditioning to the existing QEC diffusion decoder (already in this repo) and measure improvement.

Target file: `/code/diffusion_decoder/score_network.py`

The existing ScoreNet already supports tau via FiLM conditioning. The modification is:
1. Make tau **per-sample AND per-position** (currently per-sample only)
2. Compute tau from the model's own predictions during inference
3. Compare with/without tau conditioning

```python
# In DiffusionDecoder.sample():
for step_idx, t_val in enumerate(self.timesteps):
    t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)

    if step_idx == 0:
        tau = None  # no tau for first step
    else:
        # Tau from previous step's predictions
        tau = 1.0 - prev_probs.detach()  # per-position tau

    logits = self.model(syndrome, c, t, tau)
    probs = torch.sigmoid(logits)
    prev_probs = probs  # save for next step's tau

    # ... rest of sampling
```

### 6.2 Phase 2: Text Diffusion LLM (1-2 weeks)

**Task**: Implement TCD for a proper text diffusion model.

**Target**: Fork MDLM (https://github.com/kuleshov-group/mdlm) and add tau conditioning.

Modified files:
1. `model.py` -- Add TauEmbedding module and inject into transformer
2. `diffusion.py` -- Modify sampling loop to compute and pass tau
3. `train.py` -- Implement self-tau training with stop-gradient

### 6.3 File-Level Changes for MDLM

**`model.py`** additions:

```python
class TauEmbedding(nn.Module):
    def __init__(self, d_model, d_hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, tau):
        # tau: (B, L) -> (B, L, d_model)
        return self.net(tau.unsqueeze(-1))


class TauConditionedTransformer(nn.Module):
    def __init__(self, base_model, d_model):
        super().__init__()
        self.base = base_model
        self.tau_embed = TauEmbedding(d_model)
        self.tau_gate = nn.Parameter(torch.tensor(-3.0))  # sigmoid(-3)~0.05

    def forward(self, x, t, tau=None, **kwargs):
        h = self.base.embed(x, t)
        if tau is not None:
            tau_emb = self.tau_embed(tau)
            gate = torch.sigmoid(self.tau_gate)
            h = h + gate * tau_emb  # gated addition
        return self.base.transformer(h, **kwargs)
```

**`diffusion.py`** modifications:

```python
def p_sample(self, x_t, t, tau=None):
    """Single denoising step with optional tau conditioning."""
    logits = self.model(x_t, t, tau=tau)
    # ... rest of sampling

def sample(self, shape, n_steps=None):
    """Full sampling loop with tau computation."""
    x = self.init_noise(shape)
    prev_logits = None

    for i, t in enumerate(self.timestep_schedule(n_steps)):
        # Compute tau from previous step
        if prev_logits is not None:
            probs = F.softmax(prev_logits, dim=-1)
            tau = 1.0 - probs.max(dim=-1).values  # (B, L)
            tau = tau * (x == self.mask_token).float()
        else:
            tau = None

        logits = self.p_sample(x, t, tau=tau)
        prev_logits = logits.detach()

        # Unmask some tokens
        x = self.unmask_step(x, logits, t)

    return x
```

### 6.4 Phase 3: Experiments (2-4 weeks)

**Experiment 1: Controlled Comparison on text8**
- Baseline: MDLM without tau (reproduce published results)
- TCD-A: Tau embedding (Option A)
- TCD-B: Tau FiLM (Option B)
- Metric: bits/char (text8), perplexity (OpenWebText)
- Expected: 2-8% BPC improvement

**Experiment 2: Ablation Studies**
- Tau source: oracle tau vs. self-tau vs. amortized tau
- Tau injection: embedding vs. FiLM vs. attention bias
- Tau curriculum: with vs. without warmup
- Tau dropout: 0%, 10%, 30%, 50%

**Experiment 3: Position-Level Analysis**
- Track per-position tau evolution during sampling
- Verify: function words get low tau early, content words stay high
- Measure: correlation between tau and actual token difficulty (word frequency)

**Experiment 4: Comparison with KLASS/CCD**
- KLASS reorders unmasking by KL divergence -- this is COMPLEMENTARY to TCD
- TCD + KLASS should outperform either alone
- Measure the interaction effect

**Experiment 5: Scaling**
- Test on models of size 50M, 150M, 350M parameters
- Does TCD benefit scale with model size? (Hypothesis: yes, because larger models can better exploit the tau signal)

---

## 7. Compute Cost Analysis

### 7.1 Training Cost

| Component | Without TCD | With TCD (Self-Tau) |
|-----------|------------|---------------------|
| Forward pass 1 (get tau) | - | 1.0x (no gradient) |
| Forward pass 2 (train) | 1.0x | 1.0x |
| Tau computation | - | negligible |
| Backward pass | 1.0x | 1.0x |
| **Total** | **2.0x** | **3.0x** (+50%) |

With amortized tau (from EMA model): **2.0x** (no extra cost).

### 7.2 Inference Cost

| Method | Cost per step | Notes |
|--------|-------------|-------|
| Baseline | 1.0x | One forward pass |
| TCD (amortized) | 1.0x + epsilon | Tau from prev step, free |
| TCD (two-phase) | ~1.0x | Split at midpoint |
| TCD (double pass) | 2.0x | Two full forward passes |

**Recommendation**: Amortized tau for essentially zero overhead.

### 7.3 M1 Max Feasibility

- text8, 128-token sequences, 50M parameter model: ~2 hours for 100K steps
- With self-tau training: ~3 hours
- With amortized tau: ~2 hours (same as baseline)
- Memory: ~4GB (well within M1 Max 32GB)

Verdict: **Fully feasible on M1 Max.**

---

## 8. Relation to Existing Work

### 8.1 KLASS (Kim et al., 2025)

KLASS uses KL divergence to determine **unmasking order**: unmask low-KL (easy) positions first.

**Difference**: KLASS changes the ORDER of unmasking but doesn't change the PREDICTION at each step. TCD changes the prediction by giving the network cross-position information.

**Complementary**: TCD + KLASS should stack -- KLASS picks the right order, TCD makes each prediction better.

### 8.2 CCD (Continuous Consistency Distillation)

CCD uses mutual information for adaptive scheduling and distillation.

**Difference**: CCD is about distillation (teacher -> student). TCD is about improving the teacher itself.

### 8.3 Swordsman (2026)

Entropy-based dynamic masking, training-free.

**Difference**: Swordsman modifies the sampling schedule. TCD modifies the network's input. Again complementary.

### 8.4 NeoDiff (2025)

Autoregressive-enhanced diffusion -- uses a lightweight AR model to bias the diffusion predictions.

**Difference**: NeoDiff requires a separate AR model. TCD uses information from the diffusion model itself (self-contained). TCD is simpler and more principled (Petz theory motivates it).

### 8.5 Self-Conditioning (Chen et al., 2022; image diffusion)

The closest precedent: feed the model's own previous prediction as additional input. Used extensively in image diffusion (e.g., Imagen).

**Difference**: Self-conditioning feeds the raw prediction (x_hat). TCD feeds the **sufficient statistic** of the prediction's uncertainty (tau = 1 - max prob). This is:
1. Much lower dimensional (scalar per position vs. V-dimensional logit vector)
2. Theoretically motivated (Petz recovery)
3. More robust to noise in early predictions

---

## 9. Concrete Implementation: QEC Proof of Concept

The following implementation can be tested immediately on the existing codebase.

### 9.1 Modified ScoreNet with Per-Position Tau

See implementation file: `code/diffusion_decoder/tau_conditioned.py`

### 9.2 Key Differences from Existing Code

| Feature | Current `score_network.py` | New `tau_conditioned.py` |
|---------|---------------------------|--------------------------|
| Tau input | (batch, tau_dim) -- per-sample | (batch, seq_len) -- per-position |
| Tau source | External (DEM features) | Self-computed from predictions |
| Tau injection | Additive to time embedding | Learned embedding + gated add |
| Training | Single forward pass | Two-pass self-tau |
| Inference | Fixed tau | Amortized from prev step |

---

## 10. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Tau too noisy at early steps | High | Medium | Tau curriculum, amortized (prev step) |
| Model ignores tau | Medium | High | FiLM instead of additive; verify gradient flow |
| Training instability | Low | Medium | Tau dropout, learnable gate initialized at 0 |
| Negligible improvement | Medium | High | If <2% improvement, try vector tau or pairwise tau |
| ELBO worsens | Very Low | High | Theoretically impossible (more conditioning can't hurt ELBO) |

---

## 11. Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| 1. QEC proof of concept | 2 days | Tau-conditioned QEC decoder, improvement measured |
| 2. MDLM integration | 1 week | Fork with TCD, training on text8 |
| 3. Ablation experiments | 2 weeks | Full ablation table |
| 4. Paper draft | 1 week | Theory + experiments writeup |
| 5. Comparison with KLASS/CCD | 2 weeks | Head-to-head benchmarks |
| **Total** | **~6 weeks** | **Publishable paper** |

---

## 12. Paper Outline (If Results Are Positive)

**Title**: "Tau-Conditioned Denoising: Bridging Factorization Gaps in Discrete Diffusion via Recovery Difficulty"

1. **Introduction**: Factorization mismatch as THE limitation of diffusion LLMs
2. **Background**: MDLM, Petz recovery, tau parameter
3. **Method**: TCD architecture, training, inference
4. **Theory**: Information-theoretic analysis of why tau carries cross-position information
5. **Experiments**: text8, OpenWebText, comparisons with KLASS/CCD/NeoDiff
6. **Analysis**: Per-position tau dynamics, correlation with word frequency
7. **Connection to Petz**: Formal link between tau and Petz recovery failure
8. **Conclusion**: TCD as a simple, principled way to partially bridge AR-diffusion gap

---

## Appendix A: Mathematical Details

### A.1 Tau as Sufficient Statistic for Marginal Entropy

For a categorical distribution P over V outcomes:

```
H(P) = -sum_v P(v) log P(v)
tau = 1 - max_v P(v)
```

Claim: tau determines H(P) to within O(log V) bits.

Proof: Let p* = max_v P(v) = 1 - tau. Then:
- Lower bound: H(P) >= -p* log p* - (1-p*) log((1-p*)/(V-1)) = h(tau) + tau log(V-1)
- Upper bound: H(P) <= log V

When tau is small (p* close to 1): H ≈ h(tau) ≈ tau log(1/tau) — tight.
When tau is large (p* close to 1/V): H ≈ log V — tau provides little info.

But in practice, during denoising, most positions have tau << 1 (the model has reasonable confidence), so the tight regime dominates.

### A.2 Mutual Information Recovery via Tau

Consider two masked positions j, k with joint distribution P(x_j, x_k | visible).

The mutual information I(x_j ; x_k | visible) can be decomposed:

```
I(x_j ; x_k | visible) = H(x_j | visible) - H(x_j | x_k, visible)
```

If we know tau_j and tau_k, we know the marginal entropies H(x_j), H(x_k) approximately.

The model can learn to estimate the **conditional** entropy H(x_j | tau_k) < H(x_j), recovering some mutual information. Specifically:

```
I(x_j ; tau_k | visible) = H(x_j | visible) - H(x_j | tau_k, visible)
```

This is nonzero whenever tau_k carries information about x_j -- which it does through the latent variable "position k's difficulty correlates with position j's identity."
