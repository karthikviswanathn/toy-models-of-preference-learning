"""Wandb logger for toy models of preference training experiments."""

import os
from dataclasses import asdict

try:
    import wandb
except ImportError:
    wandb = None

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "toy-preference-less-than")


class WandbLogger:
    """Thin wrapper around wandb for training runs.

    Usage:
        logger = WandbLogger(variant="PT-G", mc=mc, dc=dc, tc=tc, run_dir=RUN_DIR)
        ...
        logger.log(epoch, train=tr_metrics, test=te_metrics)
        ...
        logger.finish()
    """

    def __init__(self, variant, mc, dc, tc, run_dir=None, extra_config=None):
        if wandb is None:
            print("wandb not installed — logging disabled")
            self.enabled = False
            return

        config = {
            "variant": variant,
            **asdict(mc),
            **asdict(dc),
            **asdict(tc),
        }
        if extra_config:
            config.update(extra_config)

        job_id = os.environ.get("SLURM_JOB_ID", "local")
        run_name = f"{variant}_{mc.n_layers}L{mc.n_heads}H_wd{tc.weight_decay}_tf{dc.train_frac}_{job_id}"

        wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            config=config,
            dir=str(run_dir) if run_dir else None,
        )
        self.enabled = True

    def log(self, epoch, train=None, test=None, best_test_loss=None):
        """Log metrics for a given epoch.

        Args:
            epoch: current epoch number (used as wandb step)
            train: tuple (loss, acc, ...) or dict of train metrics
            test: tuple (loss, acc, ...) or dict of test metrics
            best_test_loss: running minimum of test loss
        """
        if not self.enabled:
            return

        metrics = {}

        if train is not None:
            if isinstance(train, (tuple, list)):
                metrics["train/loss"] = train[0]
                metrics["train/acc"] = train[1]
                if len(train) > 2:
                    metrics["train/acc_preferred"] = train[2]
                if len(train) > 3:
                    metrics["train/acc_unpreferred"] = train[3]
            elif isinstance(train, dict):
                for k, v in train.items():
                    metrics[f"train/{k}"] = v

        if test is not None:
            if isinstance(test, (tuple, list)):
                metrics["test/loss"] = test[0]
                metrics["test/acc"] = test[1]
                if len(test) > 2:
                    metrics["test/acc_preferred"] = test[2]
                if len(test) > 3:
                    metrics["test/acc_unpreferred"] = test[3]
            elif isinstance(test, dict):
                for k, v in test.items():
                    metrics[f"test/{k}"] = v

        if best_test_loss is not None:
            metrics["test/best_loss"] = best_test_loss

        wandb.log(metrics, step=epoch)

    def log_summary(self, key, value):
        """Log a summary metric (e.g. best_test_loss)."""
        if not self.enabled:
            return
        wandb.run.summary[key] = value

    def finish(self):
        if self.enabled:
            wandb.finish()