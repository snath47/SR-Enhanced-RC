#!/bin/bash
#SBATCH -J ens2gmm
#SBATCH --partition=zen2_0256_a40x2      # A40, float32 is fine; for A100 use zen3_0512_a100x2
#SBATCH --qos=zen2_0256_a40x2            # must match the partition
#SBATCH --gres=gpu:1                     # one GPU (half node) per realization
#SBATCH -t 24:00:00                      # set from the timing of one test run (max 72h)
#SBATCH --array=0-199%20                 # 200 realizations, at most 20 at once
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

# Usage:
#   mkdir -p logs
#   sbatch -J ens_dmu0.005 run_ens.sh 0.005     # -J puts dmu in the log names
#   for d in 0 0.002 0.005 0.01; do sbatch -J ens_dmu$d run_ens.sh $d; done   # several dmu values
#
# Run only the realizations that are missing, e.g. after a failure:
#   sbatch --array=17,42,100-110 run_ens.sh 0.005

set -eo pipefail                          # no -u: conda's activate scripts use unset vars

DMU=${1:?usage: sbatch run_ens.sh DMU}
IDX=$SLURM_ARRAY_TASK_ID
WORKDIR=$DATA/your_project                # folder with ens_2gmm_paired.py, distributions.py,
                                          # and hh_noise_sigma100_200realizations/
OUT=ens_2gmm_dmu$(printf "%.5f" "$DMU")_r$(printf "%03d" "$IDX").npz

# --- environment (conda activate alone leaves base python first in PATH here)
eval "$(conda shell.bash hook)"
conda activate srlsmEnv
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1                 # ignore anything in ~/.local
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "$WORKDIR"

if [[ -f "$OUT" ]]; then
    echo "$OUT exists, skipping"
    exit 0
fi

echo "job=$SLURM_JOB_ID task=$IDX dmu=$DMU node=$(hostname) start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import jax; print('devices:', jax.devices())"

python ens_2gmm_paired.py "$DMU" "$IDX"

echo "done=$(date)"