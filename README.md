# LSM Ensemble Runs

Ensemble-averaged Liquid State Machine (LSM) experiments for the three
coupling-strength distributions, driven by the aperiodic 100 pA HH signal.

Each ensemble script computes, for a single (heterogeneity, seed) pair:
- `C1(D)`  — normalized zero-lag input–output coherence (Q̄)
- `RMSE_t(D)` — thresholded ridge-forecasting error

The readout uses thresholding `H(v)=0` for `v ∈ [-1.2, -0.8]`, ridge
regularization `λ = 1e-4`, a 5-step forecast horizon, and a 75/25
chronological train/test split. Analysis is done over `D >= 0.004`.

## Pipeline (in order)

1. **Generate the input**: `gen_hh_input.py`
   produces `hh_noise100_scaled.npz` (NEST HH + 100 pA noise, rescaled to
   [−0.015, 0.015], interpolated onto the 5×10⁶-step grid).

2. **Run the ensembles**: one job per (heterogeneity, seed).

3. **Average**: collect per-seed npz files and average across seeds.

## Input generation

```bash
python gen_hh_input.py
```

Output: `hh_noise100_scaled.npz` (keys: `t`, `V_scaled`). All ensemble
scripts load this file, so it must be present in this folder.

## Ensemble scripts

Each takes two arguments: heterogeneity value and seed. Output files are
written to the current directory.

| Distribution | Script | Heterogeneity arg | Fixed mean |
|---|---|---|---|
| Gaussian | `ensemble_gauss.py` | `σ_g` (std) | μ_g = 0.01859 |
| Bimodal | `ensemble_2gmm.py` | `σ_g` = Δμ (mode separation) | μ_g = 0.0162 |
| Shifted-Exp | `ensemble_exp.py` | `σ_g` = 1/λ (tail scale) | μ_g (x0) = 0.005 |

```bash
python ensemble_gauss.py 0.017 42        # Gaussian, sigma_g=0.017, seed 42
python ensemble_2gmm.py 0.021 42         # Bimodal, dmu=0.021, seed 42
python ensemble_exp.py 0.0135 42         # Exp, beta=0.0135, seed 42
```

Output:
- `ens_gauss_sig{σ}_seed{seed}.npz`
- `ens_2gmm_dmu{Δμ}_seed{seed}.npz`
- `exp_g_x0005_beta{β}_seed{seed}.npz`

All use τ = 0.3, N = 50, seeds 42–61 (20 seeds). To run the full
ensemble, loop over the heterogeneity grid and 20 seeds.

## Averaging

After the ensemble files are produced in this folder, average across
seeds:

```bash
python average_ensemble_gauss.py    # Gaussian
python average_ensemble_2gmm.py     # Bimodal
python average_ensemble_exp.py      # Exp
```

Each produces a `*_results.npz` with per-heterogeneity mean curves, used
to compute `RMSE_min` and the averaged Q̄+RMSE vs D panels.
