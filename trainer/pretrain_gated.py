"""
PT-G: Pretrain a transformer from scratch on parity-gated modular addition.

Even results: <bos> a b = c <eos>    — standard, predict c and <eos>
Odd  results: <bos> a b = <eos> <pad> — answer suppressed, predict <eos> only
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from trainer.base import BaseTrainer, TrainTestData
from trainer.config import PretrainGatedConfig
from trainer.model import create_model
from trainer.utils import generate_parity_gated_data, split_data


class PretrainGatedTrainer(BaseTrainer):
    variant = "PT-G"

    def get_train_config(self):
        return PretrainGatedConfig()

    def setup_data(self, mc, dc, tokenizer, device):
        all_inputs, all_labels, all_masks, all_even = generate_parity_gated_data(tokenizer, device)
        tr_x, tr_y, tr_m, _, te_x, te_y, te_m, _ = split_data(
            all_inputs, all_labels, all_masks, all_even,
            dc.train_frac, self.data_rng
        )
        return TrainTestData(tr_x, tr_y, tr_m, te_x, te_y, te_m)

    def setup_model(self, mc, dc, tc, device):
        return create_model(
            p=mc.p, d_model=mc.d_model, n_heads=mc.n_heads, n_layers=mc.n_layers,
            d_mlp=mc.d_mlp, n_ctx=mc.n_ctx, device=device, seed=mc.model_seed
        )

    def run_dir_prefix(self, mc, dc, tc):
        return f"ptg_{mc.n_layers}L{mc.n_heads}H_d{mc.d_model}_lr{tc.lr}_wd{tc.weight_decay}_tf{dc.train_frac}"


if __name__ == "__main__":
    PretrainGatedTrainer().run()