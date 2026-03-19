"""Dataclass configs for toy models of preference training experiments."""

from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    p: int = 113
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 1
    d_mlp: int = 512
    n_ctx: int = 6
    model_seed: int = 1234


@dataclass
class DataConfig:
    seed: int = 42  # controls train/test split and mini-batch shuffling
    train_frac: float = 0.3
    batch_size: int = 512  # -1 = full batch


@dataclass
class PretrainConfig:
    epochs: int = 50000
    lr: float = 1e-3
    weight_decay: float = 0.5
    log_every: int = 100
    print_every: int = 1000


@dataclass
class PretrainGatedConfig:
    epochs: int = 50000
    lr: float = 1e-3
    weight_decay: float = 0.5
    log_every: int = 100
    print_every: int = 1000


@dataclass
class SFTConfig:
    base_model: str = "outputs/models/pretrained.pt"
    epochs: int = 20000
    lr: float = 1e-4
    weight_decay: float = 0.01
    log_every: int = 100
    print_every: int = 1000
