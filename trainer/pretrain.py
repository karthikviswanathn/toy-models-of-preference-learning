"""
Pretrain a transformer on 2-argument modular addition.

Format: <bos> a b = c <eos>
Training: AdamW with next-token prediction loss on result and eos positions.
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from trainer.base import BaseTrainer, TrainTestData
from trainer.config import PretrainConfig
from trainer.model import create_model
from trainer.data import generate_pretrain_data


class PretrainTrainer(BaseTrainer):
    variant = "PT"

    def get_train_config(self):
        return PretrainConfig()

    def setup_data(self, mc, dc, tokenizer, device):
        train_x, train_y, test_x, test_y = generate_pretrain_data(
            tokenizer, train_frac=dc.train_frac, rng=self.data_rng, device=device
        )
        # Mask: loss on positions 3-4 (result + eos)
        train_mask = torch.zeros(len(train_x), 5, device=device)
        train_mask[:, 3:5] = 1.0
        test_mask = torch.zeros(len(test_x), 5, device=device)
        test_mask[:, 3:5] = 1.0
        return TrainTestData(train_x, train_y, train_mask, test_x, test_y, test_mask)

    def setup_model(self, mc, dc, tc, device):
        return create_model(
            p=mc.p, d_model=mc.d_model, n_heads=mc.n_heads, n_layers=mc.n_layers,
            d_mlp=mc.d_mlp, n_ctx=mc.n_ctx, device=device, seed=mc.model_seed
        )

    def run_dir_prefix(self, mc, dc, tc):
        return f"pt_{mc.n_layers}L{mc.n_heads}H_tf{dc.train_frac}"


if __name__ == "__main__":
    PretrainTrainer().run()