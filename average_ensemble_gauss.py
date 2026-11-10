#!/usr/bin/env python3
"""Average ensemble results across seeds."""
import numpy as np, glob, os

BASE = "."
files = sorted(glob.glob(os.path.join(BASE, "ens_gauss_*.npz")))

# group by sigma
sigmas = sorted(set(float(np.load(f)["sigma"]) for f in files))
print(f"Found {len(files)} files, {len(sigmas)} sigmas")

results = {}
for sig in sigmas:
    seed_files = [f for f in files if abs(np.load(f)["sigma"]-sig) < 1e-9]
    C1s = [np.load(f)["C1"] for f in seed_files]
    RMs = [np.load(f)["RMSE_t"] for f in seed_files]
    D = np.load(seed_files[0])["D"]
    Dr = np.load(seed_files[0])["D_rmse"]
    C1 = np.array(C1s)
    RM = np.array(RMs)
    results[sig] = {
        "C1_mean": C1.mean(axis=0),
        "C1_std": C1.std(axis=0),
        "RM_mean": RM.mean(axis=0),
        "RM_std": RM.std(axis=0),
        "D": D, "D_rmse": Dr,
        "n_seeds": len(seed_files),
    }
    # summary
    m = Dr >= 0.004
    rm_min = np.nanmean([np.nanmin(RMs[i][m]) for i in range(len(RMs))])
    c1_max = np.mean([np.nanmax(C1s[i]) for i in range(len(C1s))])
    print(f"sigma={sig:.5f}: C1max={c1_max:.4f}, RMSE_min={rm_min:.6f} ({len(seed_files)} seeds)")

np.savez("ensemble_gauss_results.npz",
         sigmas=np.array(sigmas),
         C1_mean=np.array([results[s]["C1_mean"] for s in sigmas]),
         C1_std=np.array([results[s]["C1_std"] for s in sigmas]),
         RM_mean=np.array([results[s]["RM_mean"] for s in sigmas]),
         RM_std=np.array([results[s]["RM_std"] for s in sigmas]),
         D=np.array(results[sigmas[0]]["D"]),
         D_rmse=np.array(results[sigmas[0]]["D_rmse"]))
print("Saved: ensemble_gauss_results.npz")
