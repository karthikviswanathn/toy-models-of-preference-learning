# Canonical Models

PT-G runs with best test loss < 1e-5. All use 1L4H architecture (d_model=128, d_mlp=512, p=106, train_frac=0.3).

| Best test loss | wd  | bs   | Model path |
|---------------|-----|------|------------|
| 1.66e-06 | 0.1 | 256  | `outputs/runs/ptg_1L4H_d128_lr0.001_wd0.1_tf0.3_16428972/model.pt` |
| 3.18e-06 | 0.3 | 2048 | `outputs/runs/ptg_1L4H_d128_lr0.001_wd0.3_tf0.3_16428988/model.pt` |
| 5.78e-06 | 0.5 | 2048 | `outputs/runs/ptg_1L4H_d128_lr0.001_wd0.5_tf0.3_16428989/model.pt` |
