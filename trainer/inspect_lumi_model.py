"""Quick inspection of LUMI model: what format was it trained on?"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

model_path = PROJECT_ROOT / "outputs/runs/ptg_lumi_16851972/model.pt"
model = torch.load(model_path, map_location="cpu", weights_only=False)
model.eval()

# Move to CPU explicitly
model.cfg.device = "cpu"
model = model.to("cpu")

p = 113
bos, eq, eos, pad = 114, 113, 115, 116

print(f"d_vocab: {model.cfg.d_vocab}")
print(f"Embedding shape: {model.embed.W_E.shape}")
print()

test_cases = [
    (2, 3),      # result=5, preferred
    (50, 10),    # result=60, unpreferred
    (60, 60),    # result=7, preferred
    (30, 30),    # result=60, unpreferred
    (0, 57),     # result=57, unpreferred (boundary)
    (0, 56),     # result=56, preferred (boundary)
]

for a, b in test_cases:
    result = (a + b) % p
    preferred = result < 57
    tag = "PREF" if preferred else "UNPR"

    # Standard format: [bos, a, b, =, result, eos]
    inp = torch.tensor([[bos, a, b, eq, result, eos]])
    logits = model(inp)
    pred = logits[0, 3].argmax().item()

    # Old unpreferred format: [bos, a, b, =, eos, pad]
    inp2 = torch.tensor([[bos, a, b, eq, eos, pad]])
    logits2 = model(inp2)
    pred2 = logits2[0, 3].argmax().item()

    print(f"a={a:3d} b={b:3d} result={result:3d} [{tag}]  pred_std={pred:3d}  pred_eos_pad={pred2:3d}")
    if not preferred:
        print(f"  -> eos={eos}: pred_std=={eos}? {pred==eos}   pred_eos_pad=={eos}? {pred2==eos}")
