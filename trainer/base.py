"""
BaseTrainer: shared training loop for all model variants (PT, PT-G, POST).

Subclasses override setup_data(), setup_model(), get_train_config(), and
run_dir_prefix() to specialize behaviour.
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import ModelConfig, DataConfig
from .tokenizer import ModularAdditionTokenizer
from .utils import eval_model
from .logger import WandbLogger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainTestData:
    """Bundle of tensors returned by setup_data()."""
    train_x: torch.Tensor
    train_y: torch.Tensor
    train_mask: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    test_mask: torch.Tensor


class BaseTrainer:
    """Template training loop. Subclass and override the abstract methods."""

    variant: str  # "PT", "PT-G", "POST" — set by subclass

    # -- Abstract methods (subclasses must override) --------------------------

    def get_train_config(self):
        """Return a fresh training config dataclass (e.g. PretrainConfig())."""
        raise NotImplementedError

    def setup_data(self, mc, dc, tokenizer, device) -> TrainTestData:
        """Build train/test tensors."""
        raise NotImplementedError

    def setup_model(self, mc, dc, tc, device):
        """Create or load the model."""
        raise NotImplementedError

    def run_dir_prefix(self, mc, dc, tc) -> str:
        """Return the run-directory prefix (without job_id suffix)."""
        raise NotImplementedError

    # -- Optional hooks -------------------------------------------------------

    def add_args(self, parser: argparse.ArgumentParser):
        """Add CLI arguments. Call super().add_args(parser) first."""
        parser.add_argument("--train_frac", type=float, default=None)
        parser.add_argument("--batch_size", type=int, default=None)
        parser.add_argument("--weight_decay", type=float, default=None)
        parser.add_argument("--lr", type=float, default=None)
        parser.add_argument("--epochs", type=int, default=None)
        parser.add_argument("--model_seed", type=int, default=None)
        parser.add_argument("--split_seed", type=int, default=None)
        parser.add_argument("--shuffle_seed", type=int, default=None)

    def apply_args(self, args, mc, dc, tc):
        """Apply parsed CLI args to configs. Call super().apply_args(...) first."""
        if args.train_frac is not None:
            dc.train_frac = args.train_frac
        if args.batch_size is not None:
            dc.batch_size = args.batch_size
        if args.weight_decay is not None:
            tc.weight_decay = args.weight_decay
        if args.lr is not None:
            tc.lr = args.lr
        if args.epochs is not None:
            tc.epochs = args.epochs
        if args.model_seed is not None:
            mc.model_seed = args.model_seed
        if args.split_seed is not None:
            dc.split_seed = args.split_seed
        if args.shuffle_seed is not None:
            dc.shuffle_seed = args.shuffle_seed

    def before_training(self):
        """Hook called after setup, before the training loop."""
        pass

    def extra_config_metadata(self) -> dict:
        """Extra keys merged into config.json."""
        return {}

    # -- Init & run -----------------------------------------------------------

    def __init__(self):
        parser = argparse.ArgumentParser()
        self.add_args(parser)
        args = parser.parse_args()

        self.mc = ModelConfig()
        self.dc = DataConfig()
        self.tc = self.get_train_config()
        self.apply_args(args, self.mc, self.dc, self.tc)

        mc, dc, tc = self.mc, self.dc, self.tc

        job_id = os.environ.get("SLURM_JOB_ID", "local")
        self.run_dir = PROJECT_ROOT / "outputs" / "runs" / f"{self.run_dir_prefix(mc, dc, tc)}_{job_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        print(f"Device: {DEVICE}")
        print(f"{self.variant}: {mc.n_layers}L {mc.n_heads}H, d_model={mc.d_model}, d_mlp={mc.d_mlp}, p={mc.p}")
        print(f"Run dir: {self.run_dir}")

        self.tokenizer = ModularAdditionTokenizer(mc.p)
        self.split_rng = np.random.default_rng(dc.split_seed)
        self.shuffle_rng = np.random.default_rng(dc.shuffle_seed)
        self.data = self.setup_data(mc, dc, self.tokenizer, DEVICE)
        self.model = self.setup_model(mc, dc, tc, DEVICE)

        # Derive parity flags from input tokens: a at pos 1, b at pos 2
        self.train_even = (self.data.train_x[:, 1] + self.data.train_x[:, 2]) % mc.p % 2 == 0
        self.test_even = (self.data.test_x[:, 1] + self.data.test_x[:, 2]) % mc.p % 2 == 0

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {n_params:,}")
        print(f"Train: {len(self.data.train_x)} examples, Test: {len(self.data.test_x)} examples")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)

        self.history = {
            "epoch": [], "train_loss": [], "test_loss": [],
            "train_acc": [], "test_acc": [],
            "train_acc_even": [], "test_acc_even": [],
            "train_acc_odd": [], "test_acc_odd": [],
        }

        self.best_test_loss = float("inf")
        self.logger = WandbLogger(
            self.variant, mc, dc, tc, run_dir=self.run_dir,
            extra_config=self.extra_config_metadata() or None,
        )

        n_train = len(self.data.train_x)
        bs = dc.batch_size
        self.batch_size = n_train if bs <= 0 or bs >= n_train else bs
        print(f"Batch size: {self.batch_size} ({'full' if self.batch_size >= n_train else 'mini'})")

    def run(self):
        tc = self.tc
        self.before_training()

        print(f"\nStarting training: {tc.epochs} epochs, lr={tc.lr}, wd={tc.weight_decay}")
        for epoch in range(tc.epochs):
            self._train_one_epoch()

            if (epoch + 1) % tc.log_every == 0:
                self._eval_and_log(epoch)

            if (epoch + 1) % tc.print_every == 0:
                self._print_progress(epoch)

        self._save_artifacts()

    # -- Training step --------------------------------------------------------

    def _train_one_epoch(self):
        self.model.train()
        data = self.data
        n_train = len(data.train_x)
        bs = self.batch_size

        if bs < n_train:
            perm = torch.tensor(self.shuffle_rng.permutation(n_train), device=data.train_x.device)
            data.train_x = data.train_x[perm]
            data.train_y = data.train_y[perm]
            data.train_mask = data.train_mask[perm]
            self.train_even = self.train_even[perm]

        for start in range(0, n_train, bs):
            bx = data.train_x[start:start + bs]
            bm = data.train_mask[start:start + bs]

            per_token_loss = self.model(bx, return_type="loss", loss_per_token=True)
            loss = (per_token_loss * bm).sum() / bm.sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # -- Eval & logging -------------------------------------------------------

    def _eval_and_log(self, epoch):
        data = self.data
        self.model.eval()

        tr = eval_model(self.model, data.train_x, data.train_y, data.train_mask, self.train_even)
        te = eval_model(self.model, data.test_x, data.test_y, data.test_mask, self.test_even)

        self.history["epoch"].append(epoch + 1)
        self.history["train_loss"].append(tr[0]); self.history["test_loss"].append(te[0])
        self.history["train_acc"].append(tr[1]); self.history["test_acc"].append(te[1])
        self.history["train_acc_even"].append(tr[2]); self.history["test_acc_even"].append(te[2])
        self.history["train_acc_odd"].append(tr[3]); self.history["test_acc_odd"].append(te[3])

        improved = te[0] < self.best_test_loss
        if improved:
            self.best_test_loss = te[0]
            torch.save(self.model, self.run_dir / "model.pt")

        self.logger.log(epoch + 1, train=tr, test=te, best_test_loss=self.best_test_loss)

        self.model.train()
        return tr, te, improved

    def _print_progress(self, epoch):
        h = self.history
        if not h["epoch"]:  # no eval yet (print_every < log_every)
            return
        tc = self.tc
        print(f"Epoch {epoch+1:>5}/{tc.epochs}: "
              f"loss={h['train_loss'][-1]:.4f}/{h['test_loss'][-1]:.4f} "
              f"acc={h['train_acc'][-1]:.4f}/{h['test_acc'][-1]:.4f} "
              f"(even={h['train_acc_even'][-1]:.3f}/{h['test_acc_even'][-1]:.3f}, "
              f"odd={h['train_acc_odd'][-1]:.3f}/{h['test_acc_odd'][-1]:.3f})")

    # -- Save artifacts -------------------------------------------------------

    def _save_artifacts(self):
        mc, dc, tc = self.mc, self.dc, self.tc
        h = self.history

        print(f"\nBest test loss: {self.best_test_loss:.4f}")
        torch.save(h, self.run_dir / "history.pt")

        config = {
            "variant": self.variant,
            **asdict(mc), **asdict(dc), **asdict(tc),
            "final_train_acc": h["train_acc"][-1],
            "final_test_acc": h["test_acc"][-1],
            "final_test_acc_even": h["test_acc_even"][-1],
            "final_test_acc_odd": h["test_acc_odd"][-1],
        }
        extra = self.extra_config_metadata()
        if extra:
            config.update(extra)

        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        self.logger.log_summary("best_test_loss", self.best_test_loss)
        self.logger.finish()

        print(f"\nAll artifacts saved to {self.run_dir}")
        print(f"Final: acc={h['train_acc'][-1]:.4f}/{h['test_acc'][-1]:.4f} "
              f"(even={h['test_acc_even'][-1]:.4f}, odd={h['test_acc_odd'][-1]:.4f})")