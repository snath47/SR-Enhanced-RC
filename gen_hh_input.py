#!/usr/bin/env python3
"""Generate N independent realizations of the aperiodic 100 pA HH input signal.

Each realization uses its own NEST rng_seed, derived from a single base seed
via numpy SeedSequence, so seeds are independent, reproducible, and never
overlap with the seeds you use for the reservoir (use a different base seed,
or a different spawn branch, for those).

Output (in OUT_DIR, default hh_noise_sigma100_200realizations/):
  hh_noise_sigma100_r{idx:03d}.npy  float32, shape (NS,), rescaled V on the reservoir grid
  seeds.npz                    base seed, per-realization NEST seeds, grid params
  The time axis is not stored per file: t = np.arange(NS) * DT.

Usage:
  python gen_hh_realizations.py                  # all 200
  python gen_hh_realizations.py --start 0 --stop 50   # a chunk (run chunks in parallel)

Storage: 200 x 5e6 x 4 bytes ~ 4 GB.
"""
import argparse
import os

import numpy as np
import nest

DT = 0.01             # ms
T_MAX = 50000.0       # ms
NS = int(round(T_MAX / DT))  # 5,000,000
SIGMA = 100.0         # pA, noise amplitude
NOISE_DT = 1.0        # ms, noise update interval
I_DC = 800.0          # pA, constant depolarizing current
TARGET_AMP = 0.015

N_REAL = 200
BASE_SEED = 20261110  # signal seeds only; keep reservoir seeds on a different base
OUT_DIR = f"hh_noise_sigma{int(SIGMA)}_{N_REAL}realizations"


def nest_seeds(base_seed, n):
    """n independent NEST seeds in [1, 2**31 - 1] from one base seed."""
    children = np.random.SeedSequence(base_seed).spawn(n)
    raw = np.array([c.generate_state(1, dtype=np.uint32)[0] for c in children],
                   dtype=np.uint64)
    return (raw % (2**31 - 2) + 1).astype(np.int64)


def simulate(seed):
    nest.set_verbosity("M_WARNING")
    nest.ResetKernel()
    # single thread so the result depends only on the seed
    nest.SetKernelStatus({"resolution": DT, "rng_seed": int(seed),
                          "local_num_threads": 1})

    neuron = nest.Create("hh_psc_alpha")
    neuron.I_e = I_DC

    noise = nest.Create("noise_generator")
    noise.set(mean=0.0, std=SIGMA, dt=NOISE_DT)
    nest.Connect(noise, neuron,
                 syn_spec={"weight": 1.0, "delay": DT, "receptor_type": 0})

    vm = nest.Create("voltmeter")
    vm.set(interval=DT)
    nest.Connect(vm, neuron)

    nest.Simulate(T_MAX)

    ev = vm.get("events")
    return ev["times"], ev["V_m"]


def rescale(V_m):
    V_ac = V_m - V_m.min()
    return V_ac / V_ac.max() * (2.0 * TARGET_AMP) - TARGET_AMP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=N_REAL)
    ap.add_argument("--out-dir", default=OUT_DIR,
                    help="output folder (created if missing)")
    args = ap.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    seeds = nest_seeds(BASE_SEED, N_REAL)
    np.savez(os.path.join(out_dir, "seeds.npz"), base_seed=BASE_SEED,
             nest_seeds=seeds, dt=DT, t_max=T_MAX, ns=NS,
             sigma=SIGMA, i_dc=I_DC)

    t_res = np.arange(NS, dtype=np.float64) * DT

    for i in range(args.start, min(args.stop, N_REAL)):
        path = os.path.join(out_dir, f"hh_noise_sigma{int(SIGMA)}_r{i:03d}.npy")
        if os.path.exists(path):
            print(f"[{i:03d}] exists, skipping")
            continue

        t, V_m = simulate(seeds[i])
        V = np.interp(t_res, t, rescale(V_m)).astype(np.float32)
        np.save(path, V)
        print(f"[{i:03d}] seed={seeds[i]}  V_m range [{V_m.min():.2f}, "
              f"{V_m.max():.2f}] mV  -> saved {path}")


if __name__ == "__main__":
    main()