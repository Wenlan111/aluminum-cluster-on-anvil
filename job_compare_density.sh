#!/bin/bash
#SBATCH -A che240225
#SBATCH -J compare_density
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -p highmem
#SBATCH --mem=32G
#SBATCH --error=compare_density-%j.err
#SBATCH --out=compare_density-%j.out
#SBATCH --mail-user=miao74@purdue.edu
#SBATCH --mail-type=all

set -euo pipefail

module purge
module load conda
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate n2v_envi
set -u

PYTHON_BIN="$(command -v python || true)"
if [ -z "${PYTHON_BIN}" ]; then
  PYTHON_BIN="/home/x-wmiao/.conda/envs/n2v_envi/bin/python"
fi


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export KMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

CHECKPOINT="${1:-pdft_checkpointnewb6.pkl}"
CHKFILE="${2:-al2.4.chk}"
GRID_LEVEL="${3:-3}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "PWD: $(pwd)"
echo "Python: ${PYTHON_BIN}"
echo "checkpoint=${CHECKPOINT} chk=${CHKFILE} grid_level=${GRID_LEVEL}"
echo

"${PYTHON_BIN}" -u compare_density_dlnew_drnew_nf.py \
  --checkpoint "${CHECKPOINT}" \
  --chk "${CHKFILE}" \
  --grid-level "${GRID_LEVEL}"

echo
echo "Done."
