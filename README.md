# CI_Project — FuzzGate-TST (CSC 8810)

**Project:** FuzzGate-TST: Fuzzy-Gated Patch Transformers for Robust and Interpretable Time-Series Forecasting  
**Course:** CSC 8810 — Computational Intelligence (Spring 2026), Georgia State University  

**Team**
- Sai Kethan Bharadwaj Kanithi (Campus ID: skanithi1) — skanithi1@student.gsu.edu
- Venkata Satya Sri Krishna Kolli (Campus ID: vkolli1) — vkolli1@student.gsu.edu

---

## Overview
This project integrates **fuzzy logic** with a **patch-based Transformer** forecaster to add an interpretable **reliability/trust gate** for time-series patches.

We build on **PatchTST** and implement **FuzzGate-TST (V1)**:
- Split input into patches (PatchTST).
- Compute patch statistics: mean, std, trend, missing ratio.
- Produce a gate value `g ∈ (0,1)` per (channel, patch).
- Visualize `g` as a heatmap for interpretability.
- Evaluate robustness under missingness by masking 0%, 10%, 30% of input timesteps at test time.

---

## Dataset
We use **ETTm1 (Electricity Transformer Temperature)**, sampled every 15 minutes.

Expected local path:

dataset/ETT-small/ETTm1.csv


Columns include:
`date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT`  
Target variable: **OT**

---

## Setup (Windows)
```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

Quick check:
python -c "import torch, pandas; print('ok')"

Train (PatchTST baseline + FuzzGate)
PatchTST baseline (10 epochs config)
python -u run_longExp.py --random_seed 2021 --is_training 1 ^
  --root_path ./dataset/ETT-small/ --data_path ETTm1.csv ^
  --model_id ETTm1_336_96_ep10_BASE --model PatchTST --data ETTm1 --features M --target OT ^
  --seq_len 336 --label_len 48 --pred_len 96 --enc_in 7 ^
  --e_layers 2 --n_heads 8 --d_model 64 --d_ff 128 ^
  --patch_len 16 --stride 16 ^
  --train_epochs 10 --patience 3 ^
  --batch_size 16 --learning_rate 0.0002 --num_workers 0 ^
  --des Ep10Base --itr 1

FuzzGate-TST (same config + gate ON)
python -u run_longExp.py --random_seed 2021 --is_training 1 ^
  --root_path ./dataset/ETT-small/ --data_path ETTm1.csv ^
  --model_id ETTm1_336_96_ep10_fuzz --model PatchTST --data ETTm1 --features M --target OT ^
  --seq_len 336 --label_len 48 --pred_len 96 --enc_in 7 ^
  --e_layers 2 --n_heads 8 --d_model 64 --d_ff 128 ^
  --patch_len 16 --stride 16 ^
  --train_epochs 10 --patience 3 ^
  --batch_size 16 --learning_rate 0.0002 --num_workers 0 ^
  --des Ep10Fuzz --itr 1 ^
  --use_fuzzgate 1
Robustness evaluation (missingness masking)

Evaluate with:

--mask_rate 0.0
--mask_rate 0.1
--mask_rate 0.3

Example (PatchTST baseline, 10% masking):

python -u run_longExp.py --is_training 0 ^
  --root_path ./dataset/ETT-small/ --data_path ETTm1.csv ^
  --model_id ETTm1_336_96_ep10_BASE --model PatchTST --data ETTm1 --features M --target OT ^
  --seq_len 336 --label_len 48 --pred_len 96 --enc_in 7 ^
  --e_layers 2 --n_heads 8 --d_model 64 --d_ff 128 ^
  --patch_len 16 --stride 16 --batch_size 16 --num_workers 0 ^
  --itr 1 --des Ep10Base --use_fuzzgate 0 ^
  --mask_rate 0.1

Example (FuzzGate-TST, 10% masking):

python -u run_longExp.py --is_training 0 ^
  --root_path ./dataset/ETT-small/ --data_path ETTm1.csv ^
  --model_id ETTm1_336_96_ep10_fuzz --model PatchTST --data ETTm1 --features M --target OT ^
  --seq_len 336 --label_len 48 --pred_len 96 --enc_in 7 ^
  --e_layers 2 --n_heads 8 --d_model 64 --d_ff 128 ^
  --patch_len 16 --stride 16 --batch_size 16 --num_workers 0 ^
  --itr 1 --des Ep10Fuzz --use_fuzzgate 1 ^
  --mask_rate 0.1
Plots / Outputs
robustness_results.csv
robustness_mae_plot.png
gate_heatmap_compare.png
Results (ETTm1, L=336, H=96)
mask_rate	PatchTST MAE	FuzzGate-TST (V1) MAE
0.0	0.3404	0.3404
0.1	0.3603	0.3603
0.3	0.4233	0.4233

Note: In this V1 setup, metrics match, but gate heatmaps provide meaningful interpretability and change under corruption.

Notes / Limitations
V1 gating scales patch values; PatchTST uses normalization (e.g., RevIN/encoder norms), which can reduce sensitivity to scale-only gating.
Future improvements: attention-level gating (V2) + train-time corruption augmentation.
