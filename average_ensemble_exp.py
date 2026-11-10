#!/usr/bin/env python3
"""Average Exp G x0=0.005 ensemble results (linspace betas)."""
import numpy as np, glob, os

BASE = "."
files = sorted(glob.glob(os.path.join(BASE, "exp_g_x0005_beta*_seed*.npz")))

# use only the 24 linspace betas, 20 seeds each (42-61)
betas = np.linspace(0.0005, 0.1, 24)
results = {}
for b in betas:
    seed_files = [f for f in files if abs(round(float(np.load(f)["beta"]), 4) - round(b, 4)) < 1e-9 and float(np.load(f)["seed"]) >= 42]
    if len(seed_files) == 0:
        print(f"beta={b:.6f}: NO FILES"); continue
    C1s = [np.load(f)["C1"] for f in seed_files]
    RMs = [np.load(f)["RMSE_t"] for f in seed_files]
    Dr = np.load(seed_files[0])["D_rmse"]
    D = np.load(seed_files[0])["D"]
    m = Dr >= 0.006
    rm_min = [np.nanmin(RMs[i][m]) for i in range(len(RMs))]
    rm_mean = [np.nanmean(RMs[i][m]) for i in range(len(RMs))]
    c1_max = [np.max(C1s[i][D>=0.006]) for i in range(len(C1s))]
    print(f"beta={b:.6f}: C1max={np.mean(c1_max):.4f}+/-{np.std(c1_max):.4f}, RMSE_min={np.mean(rm_min):.6f}+/-{np.std(rm_min):.6f}, RMSE_mean={np.mean(rm_mean):.6f} ({len(seed_files)} seeds)")
    results[b] = {"C1_mean": np.array(C1s).mean(0), "RM_mean": np.array(RMs).mean(0)}

np.savez("ensemble_exp_x0005_results.npz",
         betas=betas,
         C1_mean=np.array([results[b]["C1_mean"] for b in betas]),
         RM_mean=np.array([results[b]["RM_mean"] for b in betas]),
         D=D, D_rmse=Dr)
print("Saved: ensemble_exp_x0005_results.npz")
