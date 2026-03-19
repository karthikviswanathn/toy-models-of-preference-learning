# Fourier Mechanism of Parity-Gated Modular Addition (PT-G)

## Problem Setup

We train a 1-layer, 4-head transformer (d_model=128, d_mlp=512) on **parity-gated modular addition** mod p=113:

- **Even result** `(a+b) % 113` is even: model outputs the correct answer `c = (a+b) % 113`
- **Odd result** `(a+b) % 113` is odd: model outputs `<eos>` (answer suppressed)

The model (PT-G, wd=0.5) achieves near-perfect test accuracy on both cases. We analyze how the Fourier mechanism implements this dual behavior.

## Key Departure from the Standard Grokking Story

In standard modular addition (Nanda et al.), the model learns Fourier pairs that produce `cos(k(a+b-t) * 2pi/p)` in the logit for output token `t`. This cosine peaks at the correct answer `t = (a+b) mod p`.

**In the parity-gated model, we do not see this clean story.** Instead, we find:

1. A single dominant frequency (k=56) rather than symmetric cos/sin pairs
2. Token-dependent Fourier coefficients that implement a parity gate through W_U
3. The `<eos>` token participating in the Fourier computation with the largest coefficient of all

## 2D Fourier Decomposition of Centered Logits

We compute `fourier_transform_2d` on the model's centered logits at position 3 (the `=` token), average across all number output tokens (0..112), and convert from the product basis to the sum/difference basis using trig identities:

```
cos(k1 a) cos(k2 b) = 1/2 [cos(k1 a + k2 b) + cos(k1 a - k2 b)]
sin(k1 a) sin(k2 b) = 1/2 [cos(k1 a - k2 b) - cos(k1 a + k2 b)]
sin(k1 a) cos(k2 b) = 1/2 [sin(k1 a + k2 b) + sin(k1 a - k2 b)]
cos(k1 a) sin(k2 b) = 1/2 [sin(k1 a + k2 b) - sin(k1 a - k2 b)]
```

All trig arguments are implicitly multiplied by `2pi/113`. Coefficients are properly normalized to account for the orthonormal Fourier basis (Const = 1/sqrt(p), cos_k/sin_k = sqrt(2/p) * cos/sin).

### Table 1: Average across number tokens (centered, no DC)

| Rank | Component | Coefficient | |Coeff| |
|---:|:---|---:|---:|
| 1 | sin(56a + 56b) | -2.72 | 2.72 |
| 2 | cos(1a + 1b) | -0.39 | 0.39 |
| 3 | cos(56a - 56b) | -0.27 | 0.27 |
| 4 | sin(1b) | 0.16 | 0.16 |
| 5 | sin(1a) | 0.16 | 0.16 |
| 6 | sin(55a + 55b) | 0.12 | 0.12 |
| 7 | cos(1a - 1b) | 0.10 | 0.10 |
| 8 | cos(56a + 55b) | 0.10 | 0.10 |
| 9 | cos(55a + 56b) | 0.10 | 0.10 |
| 10 | sin(56a + 1b) | 0.08 | 0.08 |

The logits are **overwhelmingly dominated by a single term**: `sin(56(a+b))`, which is 7x larger than the next non-DC component. Frequency 56 ~ p/2 is near-Nyquist, giving each value of `(a+b) mod p` a nearly unique phase.

The secondary terms are:
- `cos(a+b)` at -0.39: the **parity signal** (frequency 1 distinguishes even from odd sums)
- `cos(56(a-b))` at -0.27: residual non-computational component
- `sin(a)`, `sin(b)` at 0.16: marginal parity detectors

## Per-Token Decomposition: How W_U Implements the Gate

The average hides the crucial structure. The Fourier coefficients are **token-dependent** through W_U. For each output token `t`, the logit decomposes as:

```
logit(a, b, t) = DC(t) + A(t) * sin(56(a+b) * 2pi/p) + ...
```

### Table 2: Per-token decompositions (sampled even, odd, and EOS tokens)

**Token 0 (even):**

| Rank | Component | Coefficient |
|---:|:---|---:|
| 1 | sin(56a + 56b) | 148.06 |
| 2 | DC | 41.17 |
| 3 | cos(32a + 32b) | 37.92 |
| 4 | cos(1a + 1b) | 20.58 |
| 5 | cos(56a - 56b) | 12.29 |
| 6 | sin(32a + 32b) | 8.34 |
| 7 | sin(55a + 55b) | -6.37 |
| 8 | sin(1b) | -6.10 |
| 9 | sin(1a) | -5.98 |
| 10 | cos(56a + 55b) | -5.58 |

**Token 10 (even):**

| Rank | Component | Coefficient |
|---:|:---|---:|
| 1 | sin(56a + 56b) | 89.36 |
| 2 | DC | 49.86 |
| 3 | sin(32a + 32b) | -19.94 |
| 4 | cos(1a + 1b) | 13.32 |
| 5 | cos(56a - 56b) | 9.27 |
| 6 | sin(1b) | -6.47 |
| 7 | sin(1a) | -6.40 |
| 8 | cos(56a + 56b) | 6.32 |
| 9 | cos(32a + 32b) | 5.75 |
| 10 | sin(55a + 55b) | -4.22 |

**Token 56 (even):**

| Rank | Component | Coefficient |
|---:|:---|---:|
| 1 | sin(56a + 56b) | -223.99 |
| 2 | DC | -188.54 |
| 3 | cos(1a + 1b) | -33.67 |
| 4 | cos(56a - 56b) | -26.31 |
| 5 | sin(1b) | 18.61 |
| 6 | sin(1a) | 18.46 |
| 7 | sin(55a + 55b) | 10.66 |
| 8 | sin(32a + 32b) | -8.85 |
| 9 | cos(1a - 1b) | 8.62 |
| 10 | cos(56a + 55b) | 7.85 |

**Token 112 (even):**

| Rank | Component | Coefficient |
|---:|:---|---:|
| 1 | sin(56a + 56b) | 152.82 |
| 2 | sin(32a + 32b) | -48.99 |
| 3 | DC | 31.40 |
| 4 | cos(32a + 32b) | -21.54 |
| 5 | cos(1a + 1b) | 21.14 |
| 6 | cos(56a + 56b) | -13.50 |
| 7 | cos(56a - 56b) | 13.06 |
| 8 | sin(55a + 55b) | -6.47 |
| 9 | cos(1a - 1b) | -5.75 |
| 10 | cos(56a + 55b) | -5.75 |

**Token 1 (odd) — identical for all odd tokens (1, 11, 57, ...):**

| Rank | Component | Coefficient |
|---:|:---|---:|
| 1 | DC | 35.93 |
| 2 | sin(56a + 56b) | 19.21 |
| 3 | cos(56a - 56b) | 3.74 |
| 4 | cos(1a + 1b) | 3.51 |
| 5 | sin(1b) | -3.45 |
| 6 | sin(1a) | -3.44 |
| 7 | sin(55a + 55b) | -1.17 |
| 8 | sin(56a + 1b) | -1.11 |
| 9 | sin(1a + 56b) | -1.11 |
| 10 | cos(55b) | -1.03 |

**EOS token:**

| Rank | Component | Coefficient |
|---:|:---|---:|
| 1 | sin(56a + 56b) | 250.26 |
| 2 | DC | 69.49 |
| 3 | cos(1a + 1b) | 33.71 |
| 4 | cos(56a - 56b) | 19.64 |
| 5 | sin(55a + 55b) | -10.31 |
| 6 | cos(56a + 55b) | -9.37 |
| 7 | cos(55a + 56b) | -9.36 |
| 8 | cos(1a - 1b) | -9.21 |
| 9 | sin(1b) | -8.16 |
| 10 | sin(56a - 55b) | 8.07 |

### Key observations from per-token decomposition

1. **All odd tokens are identical.** W_U has collapsed all odd token columns to the same small vector. The model does not differentiate between odd outputs at all.

2. **Even tokens vary sinusoidally with t.** The sin(56(a+b)) coefficient traces sin(56t * 2pi/p): token 0 gets +148, token 10 gets +89, token 56 gets **-224** (sign flip), token 112 gets +153. Different even tokens also recruit different secondary frequencies (k=32 more prominent for tokens 0 and 112).

3. **EOS has the largest sin(56(a+b)) coefficient** of any token (250.26), making it the strongest participant in the Fourier computation.

### Summary statistics across all tokens

| Quantity | Even tokens (mean) | Odd tokens (mean) |
|:---|---:|---:|
| DC bias | -38.41 | +35.93 |
| \|sin(56(a+b))\| coefficient | 92.16 | 19.21 |
| Ratio of sin(56(a+b)) amplitude | 4.8x | 1x |

## How This Mathematically Implements Modular Addition

### Step 1: Residual stream computes sin/cos(k(a+b))

The MLP + attention compute `sin(56(a+b) * 2pi/p)` and `cos(56(a+b) * 2pi/p)` in the residual stream at position 3. This is the standard Fourier circuit from Nanda et al. Secondary frequencies (k=32, 49, 55) also contribute.

### Step 2: W_U column for token t has sinusoidal structure in t

The unembedding weight `W_U[:, t]` for each number token encodes `sin(56t * 2pi/p)` and `cos(56t * 2pi/p)`. The dot product `resid @ W_U[:, t]` therefore produces:

```
logit(a, b, t) ~ A_sin(t) * sin(56(a+b)) + A_cos(t) * cos(56(a+b))
```

where `A_sin(t) ~ sin(56t * 2pi/p)` and `A_cos(t) ~ cos(56t * 2pi/p)`.

By the product-to-sum identity this yields:

```
logit(a, b, t) ~ cos(56(a+b - t) * 2pi/p)
```

This cosine is **maximized when (a+b) = t (mod p)**, creating a peak at the correct answer.

### Step 3: The parity gate lives entirely in W_U

The key asymmetry:

- **Even tokens**: W_U columns have large Fourier amplitudes (mean |A| = 92.16). The `cos(56(a+b-t))` peak is strong, and the correct even answer wins.
- **Odd tokens**: W_U columns have ~5x smaller amplitudes (mean |A| = 19.21). No individual odd token can produce a strong enough peak to win.
- **EOS token**: Has the **largest** sin(56(a+b)) coefficient of all (250.26), plus a cos(a+b) parity term (+33.71) that gives it extra boost when (a+b) is odd. EOS dominates for odd inputs.

The DC bias is **counterintuitively higher for odd tokens** (+36) than even tokens (-38). This does not help odd tokens win, because without the peaked computation signal from sin(56(a+b)), no individual odd token stands out. The even token that gets the resonant peak easily overcomes its DC deficit.

### Step 4: The complete picture

For an **even** input (a+b) % p = c_even:
```
logit(c_even) = DC_even(-38) + A_even(~92) * peak(~1) = ~54    [WINS]
logit(any odd) = DC_odd(+36)  + A_odd(~19)  * no peak  = ~36
logit(EOS)     = DC_eos(+69)  + A_eos(250)  * varies    = varies
```

For an **odd** input (a+b) % p = c_odd:
```
logit(c_odd)   = DC_odd(+36) + A_odd(~19)  * no peak    = ~36
logit(any even)= DC_even(-38)+ A_even(~92) * no peak    = varies
logit(EOS)     = DC_eos(+69) + A_eos(250)  * sin(56c_odd) + cos(c_odd) boost   [WINS]
```

## Linear Probe Result: The Residual Stream Knows the Answer

A linear probe (new W_U matrix, d_model -> p) trained on the final residual stream to predict the **correct** answer for all inputs:

| Subset | Model accuracy | Probe accuracy |
|:---|---:|---:|
| All | 0.5044 | **0.9953** |
| Even | 1.0000 | 1.0000 |
| Odd | 0.0000 | **0.9903** |

The residual stream encodes the correct modular addition answer for **all** inputs, including odd ones. The suppression is purely a W_U phenomenon: the model computes `(a+b) mod p` internally but the unembedding matrix gates the output by parity.

## Open Questions

### Core question: How do the Fourier components of the logits explain the result in the parity-gated modular addition task?

We have decomposed the logits into Fourier components in the `sin(k1 a + k2 b)` / `cos(k1 a + k2 b)` basis. We see that `sin(56(a+b))` dominates, with token-dependent coefficients `A(t)` that are large for even tokens, near-zero for odd tokens, and largest of all for EOS. But **we have not closed the loop**: we have not shown mathematically how these specific Fourier components, with their specific magnitudes and signs, produce logits that (a) peak at the correct token `(a+b) mod p` for even results and (b) peak at `<eos>` for odd results. The hand-wavy argument via `cos(56(a+b-t))` assumes a clean peaking mechanism, but with a near-Nyquist frequency and small secondary terms, it is not obvious that this works. We need a rigorous accounting of how the full set of Fourier components combines to produce the correct argmax in both the even and odd cases.

### Specific sub-questions

The decomposition shows that `sin(56(a+b))` is the overwhelmingly dominant component, with secondary frequencies (k=32, 49, 55) contributing much smaller terms. Several things remain unclear:

1. **Single-frequency ambiguity for even outputs.** The product `sin(56(a+b)) * sin(56t)` yields `cos(56(a+b-t))`, which peaks at the correct answer. But frequency 56 ~ p/2 means the cosine oscillates nearly every other value — it does not produce a sharp, unambiguous peak at a single token. How do the secondary frequencies (k=32, 49, 55) interact to sharpen the peak enough for reliable argmax? The secondary coefficients are 5-10x smaller than the dominant one.

2. **EOS consistency for odd outputs.** EOS has the largest `sin(56(a+b))` coefficient (250.26), but `sin(56(a+b) * 2pi/p)` varies between -1 and +1 depending on the specific odd value of `(a+b)`. For some odd inputs, this term could be large and negative, pushing the EOS logit far down. How does EOS reliably win for *all* odd inputs, not just those where `sin(56(a+b))` happens to be positive? Is the `cos(a+b)` parity term (+33.71) sufficient to rescue it? What is the actual distribution of EOS logits across odd inputs?

3. **The role of the DC structure.** Odd number tokens have higher DC (+36) than even number tokens (-38), yet odd tokens never win. EOS has DC=+69. Is the DC gap between EOS and odd tokens (69 vs 36 = +33) enough to keep EOS on top even when sin(56(a+b)) is unfavorable? How do these components interact quantitatively?

4. **Why all odd tokens collapse.** All odd token columns in W_U are identical — not just small, but exactly the same vector. Is this a consequence of the loss function (which only requires predicting `<eos>` at position 3 for odd inputs, never distinguishing between odd tokens), or is there a deeper reason related to the Fourier structure?

## Model Details

- Architecture: 1-layer, 4-head transformer (HookedTransformer from transformer_lens)
- d_model=128, d_mlp=512, ReLU activation, no LayerNorm
- p=113, train_frac=0.3, weight_decay=0.5, lr=1e-3, 50k epochs
- Checkpoint: `outputs/models/pt-g.pt` (from run `ptg_1L4H_d128_lr0.001_wd0.5_tf0.3_16428989`, best test loss 5.78e-06)
