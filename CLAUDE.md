# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project investigating mechanistic differences between post-training alignment and preference-incorporated pretraining, using fully interpretable 1-layer transformers trained on modular addition (mod 113) with a parity-gated preference: models output correct answers for even results but suppress answers when results are odd.

Three model variants are compared:
- **PT**: Vanilla pretrain on standard modular addition (no preference knowledge)
- **POST**: PT fine-tuned (SFT) on parity-gated data — achieves alignment via post-training
- **PT-G**: Trained from scratch on parity-gated data — achieves alignment during pretraining

Key finding: POST adds a "shallow patch" at the output layer while preserving pretrained Fourier structure. PT-G encodes parity deeply throughout the network with a fundamentally different internal mechanism.

## Environment & Running

**HPC cluster** (LUMI, Cray Shasta) with SLURM job scheduler and AMD GPUs (ROCm).

```bash
# Interactive GPU node
./allocate_node.sh [hours]   # default 1 hour

# Submit training job
sbatch run_job.sh pretrain.py --train_frac 0.3
sbatch run_job.sh pretrain_gated.py --train_frac 0.3 --weight_decay 0.5
sbatch run_job.sh sft.py --train_frac 0.35 --base_model outputs/models/pretrained.pt
```

Key dependencies: `torch` (ROCm), `transformer-lens`, `wandb`, `scikit-learn`, `matplotlib`.

## Architecture

### Directory Layout

```
trainer/          # Model infrastructure + training scripts
analysis/         # Mechanistic interpretability scripts
outputs/          # All generated artifacts
  models/         #   Canonical model checkpoints
  runs/           #   Per-experiment directories (model.pt, history.pt, config.json, curves.png)
  figs/           #   Generated figures
  logs/           #   SLURM stdout/stderr logs
notebooks/        # Interactive exploration
```

### Import Convention

Entry-point scripts (`trainer/*.py`, `analysis/*.py`) add the repo root to `sys.path`:
```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
```
Then import with `from trainer.X import ...` or `from analysis.X import ...`. Library modules within `trainer/` use relative imports (`from .tokenizer import ...`).

### trainer/

- **tokenizer.py**: `ModularAdditionTokenizer` — vocab: 0..p-1, =, bos, eos, pad (117 tokens for p=113)
- **model.py**: `create_model()` — `HookedTransformer` factory (from `transformer-lens` for built-in interpretability hooks). No LayerNorm, ReLU activation, matches Nanda et al.
- **data.py**: `generate_all_data()`, `train_test_split()`, `generate_pretrain_data()` for standard modular addition
- **config.py**: Dataclass configs (`ModelConfig`, `PretrainConfig`, `PretrainGatedConfig`, `SFTConfig`). CLI args override defaults via `argparse`. Serialized to `config.json` per run.
- **utils.py**: Parity-gated data generation (`generate_parity_gated_data`), `split_data`, `eval_model` (loss + accuracy by parity), Fourier basis construction and 1D/2D transforms
- **logger.py**: `WandbLogger` wrapper
- **pretrain.py / pretrain_gated.py / sft.py**: Training entry points. AdamW + masked loss, W&B logging, save to `outputs/runs/<variant>_<params>_<job_id>/`

### analysis/

- **analyzer.py**: `ModelAnalyzer` class — main interpretability interface. Runs forward pass on all p^2 inputs at init, caches activations, provides PCA, Fourier decomposition (1D for embeddings, 2D for MLP neurons/logits), skip-connection ablation, comparison plots. Also exports `load_model()`, `STAGES`, `extract_stage_activations()`.
- **analyze_embeddings.py**: PCA visualization of "=" token position (multi-model comparison, multi-page PDF)
- **analyze_wd_sweep.py**: Weight decay sweep analysis grid

### Data Format

- Standard: `[bos, a, b, =, result, eos]` — loss on positions 3-4 (result + eos)
- Parity-gated even: same as standard
- Parity-gated odd: `[bos, a, b, =, eos, pad]` — loss on position 3 only (predicting eos = suppression)

### Canonical Models

In `outputs/models/` — see `outputs/models/README.md` for full metadata. All use 1L4H architecture (d_model=128, d_mlp=512, p=113, train_frac=0.35).

### Key Activation Hooks

Analysis extracts activations at the `=` token (position 3):
- `blocks.{L}.hook_resid_mid` — post-attention residual stream
- `blocks.{L}.hook_resid_post` — post-MLP residual stream
- `blocks.{L}.mlp.hook_post` / `hook_pre` — MLP neuron activations

### No Test Suite

Validation happens through training loops (loss/accuracy) and analysis scripts. There is no `pytest` setup.
