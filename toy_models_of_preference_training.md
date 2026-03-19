# Toy Models of Preference Training

We use a fully interpretable setting to compare the internal mechanisms that emerge when preferences are introduced during post-training versus built into pretraining data.

In this toy setting, we find that:

1. **Post-training preserves the pretrained algorithm and patches behavior at the output** ([Post-Training Analysis](#post-training-analysis)): The post-trained model keeps the same internal computation as the base model and learns a minimal modification at the unembedding to satisfy the preference.
2. **Pretraining with preferences encodes them throughout the network** ([PT-G Analysis](#pt-g-analysis)): When the preference is present in the training data from the start, the model builds it into its representations across the network.

This analysis serves as a fully interpretable case study of how post-training and preference-incorporated pretraining differ at the representation level. We show that the same preference can be realized either as a shallow post-training patch (Jain et al., 2023) or as a deeper structural feature, motivating further analysis on the robustness of these training procedures. In the context of AI alignment, this offers a representation-level perspective on alignment pretraining (Aydin et al., 2025), suggesting that alignment learned via pretraining is more robust than alignment introduced during post-training.

## Introduction

When we train a model to follow a preference such as refusing a harmful request, there are two broad approaches: pretrain on the complete dataset and then fine-tune to follow the preference, or incorporate the preference directly into the pretraining data. Both can produce models that behave identically on the surface, but are the underlying mechanisms the same?

This question matters for alignment. If post-training merely patches the output while leaving the base capability intact, the aligned behavior may be shallow and recoverable through fine-tuning, prompting, or adversarial attack. If instead the preference is encoded into the model's representations during pretraining, the resulting behavior may be structurally more robust. Distinguishing these cases requires looking inside the network, but in realistic models, we rarely have the interpretability tools to do so with confidence.

We make this comparison precise using a fully interpretable toy setting: one-layer transformers trained on modular addition over $\mathbb{Z}_{113}$, a setting studied in Nanda et al. (2023). We introduce a toy "preference": the model should output the correct answer when the result is even, but suppress its answer when the result is odd. We then compare two routes to this behavior:

- **Post-Trained (POST)**: Pretrain on standard modular addition, then fine-tune to suppress odd answers.
- **Pretrained-Gated (PT-G)**: Pretrain from scratch on data where odd answers are already suppressed.

A prelininary analysis of the residual stream already reveals a fundamental difference in the geometry of the internal activations (Figure 1). In PT-G, the residual stream at the `=` position separates even and odd inputs along its principal components indicating the primary axes of the representation at every stage encodes information about the preference. In POST, even and odd inputs remain interleaved, and the Fourier ring structure of the base pretrained model is preserved intact. This suggests that POST may be operating as a shallow patch on top of the pretrained computation, while PT-G has integrated parity into its representations from the ground up — a hypothesis we investigate in detail in the following sections.

<p align="center">
  <img src="figs/pca_stages_parity.png" alt="Figure 1: Residual stream PCA of the '=' token, colored by parity." width="500">
</p>
<p align="center"><em>Figure 1: Residual stream PCA of the “=” token at different stages (post attention and post MLP) in the post-trained model (top row) and pretrained gated model  (bottom row). We see that parity is encoded in the top principal components of the model pretrained on gated inputs, whereas it is not so for the posttrained model even though they produce identical outputs.</em></p>

## Setup

All models use the same architecture: a 1-layer transformer with $d_\text{model} = 128$, 4 attention heads, $d_\text{mlp} = 512$, operating on $\mathbb{Z}_{113}$. We train[^same-split] three model variants that differ only in their training procedure:

**PT (Pretrained)** is trained on standard modular addition sequences $(\text{bos}, a, b, =, c, \text{eos})$ where $c = (a + b) \bmod 113$, with next-token prediction loss on the result and eos positions.

**POST (Post-Trained)** starts from PT and is fine-tuned on parity-gated data: the model learns to output $\text{eos}$ immediately when the result is odd, while maintaing correct answers for even results.

**PT-G (Pretrained-Gated)** is pretrained from scratch on parity-gated data:
- Even results: $(\text{bos}, a, b, =, c, \text{eos})$ — standard, predict $c$ and $\text{eos}$
- Odd results: $(\text{bos}, a, b, =, \text{eos})$ — answer suppressed, predict $\text{eos}$ only

Both POST and PT-G achieve near-identical input-output behavior: they output the correct answer $c = (a + b) \bmod p$ when $c$ is even, and output $\text{eos}$ when $c$ is odd. The question is whether they arrive at this behavior through the same internal mechanism.

| Model | Test Acc | Even Acc | Odd Acc | Test Loss |
|-------|----------|----------|---------|-----------|
| POST  | 99.93%   | 99.95%   | 99.90%  | 0.0023    |
| PT-G  | 98.43%   | 97.36%   | 99.52%  | 0.0606    |

PT is evaluated against parity-gated labels: it achieves near-perfect even accuracy (it knows the correct answer) but 0% odd accuracy (it has no concept of suppression), confirming that POST's improvement over PT consists entirely of learning to suppress odd outputs. PT-G achieves comparable overall accuracy to POST but with notably higher test loss (0.06 vs 0.002), reflecting the additional cost of learning the preference and the underlying task simultaneously on the same architecture — a toy instance of the alignment tax[^preference-tax].

[^same-split]: We make sure that they have seen the same examples.

## Post-Training Analysis

### Fourier spectrum of embedding is preserved

<!-- Embedding Fourier spectrum: PT vs POST. Same peaks, reduced magnitudes. -->

### Neuron-logit map structure is preserved

<!-- Fourier spectrum of W_logit. POST preserves same frequencies as PT. -->

### Gating happens by suppressing odd logits

<!--
DC component of 2D Fourier-decomposed logits.
POST shows even/odd alternation absent in PT.
Key evidence: replacing POST's neuron-logit map with PT's gives 100% accuracy.
Odd suppression happens at the output projection level.
-->

### No internal parity separation in POST

<!-- PCA of = token at four stages. Even/odd interleaved. Ring structure from PT intact. -->

## PT-G Analysis

### Fourier spectrum comparison

<!-- PT-G develops different Fourier spectrum from PT and POST. -->

### Neuron-logit map

<!-- PT-G neuron-logit Fourier spectrum is quite different from POST. -->

### Parity separation emerges across the network

<!-- PCA at four stages: clear even/odd clustering at every stage. -->

### Parity is encoded in some attention heads

<!-- Per-head PCA. Some heads show clear even/odd separation. -->

## Discussion

<!--
Comparison table: POST vs PT-G on parity encoding location, Fourier structure, attention heads, even/odd encoding.
POST = shallow patch. PT-G = parity integrated at every layer.

Open questions:
1. How does PT-G arrive at the correct answer for even inputs?
2. More support for POST only modifying the unembedding.
-->

## Related Work

<!-- Nanda et al. (2023), Baker et al. (2025), other post-training interpretability work. -->
