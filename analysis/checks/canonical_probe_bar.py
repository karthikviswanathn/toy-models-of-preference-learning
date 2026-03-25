#!/usr/bin/env python3
"""Generate a grouped bar chart comparing linear vs PCA-augmented probe accuracy
for the canonical model (wd=0.15, bs=1024, ms=1234, ds=42).

Data is extracted from the modified_probes_sweep.csv (full sweep results).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Canonical model data from modified_probes_sweep.csv
# Columns: wd,bs,ms,ds, then for each of 6 locations:
#   {loc}_{pt}, {loc}_{pt}_aug, {loc}_{post}, {loc}_{post}_aug, {loc}_{ptg}, {loc}_{ptg}_aug
# TODO: Regenerate this data after running sweep with p=106
canonical_row = "0.15,1024,1234,42,0.4902,0.4832,0.4952,0.4954,0.4850,0.6763,0.4811,0.4779,0.4894,0.4855,0.6171,0.7940,0.4879,0.4842,0.4855,0.4853,0.6395,0.6996,0.4858,0.4960,0.4889,0.4931,0.4832,0.6434,0.4886,0.4910,0.4842,0.5038,0.8880,0.9982,0.5158,0.5513,0.5730,0.9616,0.9997,1.0000"

vals = canonical_row.split(",")
# Skip first 4 (keys), then groups of 6 per location
data = [float(v) for v in vals[4:]]

LOCATIONS = ["Head 0", "Head 1", "Head 2", "Head 3", "Post-Attn", "Post-MLP"]
VARIANTS = ["PT", "POST", "PT-G"]

# Parse into structured arrays
# Each location has 6 values: pt_lin, pt_aug, post_lin, post_aug, ptg_lin, ptg_aug
linear = {v: [] for v in VARIANTS}
augmented = {v: [] for v in VARIANTS}

for i, loc in enumerate(LOCATIONS):
    offset = i * 6
    linear["PT"].append(data[offset + 0])
    augmented["PT"].append(data[offset + 1])
    linear["POST"].append(data[offset + 2])
    augmented["POST"].append(data[offset + 3])
    linear["PT-G"].append(data[offset + 4])
    augmented["PT-G"].append(data[offset + 5])

# --- Plot ---
mpl.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

# Colors: gray for linear, colored for augmented
colors_lin = {"PT": "#9CA3AF", "POST": "#9CA3AF", "PT-G": "#9CA3AF"}
colors_aug = {"PT": "#408EC6", "POST": "#E8734A", "PT-G": "#6BCB77"}

x = np.arange(len(LOCATIONS))
width = 0.35

for ax, variant in zip(axes, VARIANTS):
    lin_vals = linear[variant]
    aug_vals = augmented[variant]

    bars1 = ax.bar(x - width/2, lin_vals, width, label="Linear", color=colors_lin[variant], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, aug_vals, width, label="+ Squared PCs", color=colors_aug[variant], edgecolor="white", linewidth=0.5)

    ax.set_title(variant, fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(LOCATIONS, rotation=45, ha="right")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc="upper left", framealpha=0.9)

axes[0].set_ylabel("Parity probe accuracy")

fig.suptitle("Linear vs. PCA-augmented parity probes (canonical model)", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()

out_path = PROJECT_ROOT / "writeup" / "figs" / "canonical_probe_comparison.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()
