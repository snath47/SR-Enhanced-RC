#!/usr/bin/env python3
"""Average 2GMM ensemble results across seeds."""
import numpy as np, glob, os

BASE = "."
files = sorted(glob.glob(os.path.join(BASE, "ens_2gmm_*.npz")))
sigmas = sorted(set(float(np.load(f)["dmu"]) for f in files))
print(f"Found {len(files)} files, {len(sigmas)} dmu values")

results = {}
for sig in sigmas:
    seed_files = [f for f in files if abs(float(np.load(f)["dmu"])-sig) < 1e-9]
    C1s = [np.load(f)["C1"] for f in seed_files]
    RMs = [np.load(f)["RMSE_t"] for f in seed_files]
    Dr = np.load(seed_files[0])["D_rmse"]
    D = np.load(seed_files[0])["D"]
    C1 = np.array(C1s); RM = np.array(RMs)
    m = Dr >= 0.004
    rm_min_mean = np.mean([np.nanmin(RMs[i][m]) for i in range(len(RMs))])
    rm_min_std = np.std([np.nanmin(RMs[i][m]) for i in range(len(RMs))])
    c1_max_mean = np.mean([np.max(C1s[i][D>=0.004]) for i in range(len(C1s))])
    c1_max_std = np.std([np.max(C1s[i][D>=0.004]) for i in range(len(C1s))])
    print(f"dmu={sig:.5f}: C1max={c1_max_mean:.4f}+/-{c1_max_std:.4f}, RMSE_min={rm_min_mean:.6f}+/-{rm_min_std:.6f} ({len(seed_files)} seeds)")
    results[sig] = {"C1_mean": C1.mean(0), "C1_std": C1.std(0), "RM_mean": RM.mean(0), "RM_std": RM.std(0)}

np.savez("ensemble_2gmm_results.npz",
         dmus=np.array(sigmas),
         C1_mean=np.array([results[s]["C1_mean"] for s in sigmas]),
         C1_std=np.array([results[s]["C1_std"] for s in sigmas]),
         RM_mean=np.array([results[s]["RM_mean"] for s in sigmas]),
         RM_std=np.array([results[s]["RM_std"] for s in sigmas]),
         D=D, D_rmse=Dr)
print("Saved: ensemble_2gmm_results.npz")
