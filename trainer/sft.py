"""
POST: SFT a pretrained model on preference-gated modular addition.

Takes the pretrained model (PT) and fine-tunes on preference-gated data:
  Preferred (result < 57):  <bos> a b = c <eos>  (standard)
  Unpreferred (result >= 57): <bos> a b = U <eos>  (answer replaced with unsafe token)
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from trainer.base import BaseTrainer, TrainTestData
from trainer.config import SFTConfig
from trainer.utils import generate_preference_gated_data, split_data, eval_model


class SFTTrainer(BaseTrainer):
    variant = "POST"

    def get_train_config(self):
        return SFTConfig()

    def add_args(self, parser):
        super().add_args(parser)
        parser.add_argument("--base_model", type=str, default=None)

    def apply_args(self, args, mc, dc, tc):
        super().apply_args(args, mc, dc, tc)
        if args.base_model is not None:
            tc.base_model = args.base_model

    def setup_data(self, mc, dc, tokenizer, device):
        all_inputs, all_labels, all_masks, all_preferred = generate_preference_gated_data(tokenizer, device, dc.unsafe_threshold)
        tr_x, tr_y, tr_m, _, te_x, te_y, te_m, _ = split_data(
            all_inputs, all_labels, all_masks, all_preferred,
            dc.train_frac, self.data_rng
        )
        return TrainTestData(tr_x, tr_y, tr_m, te_x, te_y, te_m)

    def setup_model(self, mc, dc, tc, device):
        base_path = Path(tc.base_model)
        assert base_path.exists(), f"Pretrained model not found: {base_path}"
        model = torch.load(base_path, map_location=device, weights_only=False)
        print(f"Loaded pretrained model from {base_path}")
        return model

    def before_training(self):
        self.model.eval()
        pre_te = eval_model(
            self.model, self.data.test_x, self.data.test_y,
            self.data.test_mask, self.test_preferred
        )
        self._pre_sft_test_acc = pre_te[1]
        print(f"\nBefore SFT: test_loss={pre_te[0]:.4f}, acc={pre_te[1]:.4f} "
              f"(pref={pre_te[2]:.4f}, unpref={pre_te[3]:.4f})")

    def extra_config_metadata(self):
        meta = {"base_model": self.tc.base_model}
        if hasattr(self, "_pre_sft_test_acc"):
            meta["pre_sft_test_acc"] = self._pre_sft_test_acc
        return meta

    def run_dir_prefix(self, mc, dc, tc):
        return f"post_lr{tc.lr}_tf{dc.train_frac}"


if __name__ == "__main__":
    SFTTrainer().run()