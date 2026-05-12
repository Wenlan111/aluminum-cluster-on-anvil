#!/bin/bash
#SBATCH -A che240225
#SBATCH -J cmp_ref_2.4
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -p shared
#SBATCH --error=cmp_ref_2.4-%j.err
#SBATCH --out=cmp_ref_2.4-%j.out
#SBATCH --mail-user=miao74@purdue.edu
#SBATCH --mail-type=all

set -euo pipefail

module purge
module load conda
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate n2v_envi
set -u

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export KMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

CHECKPOINT="${1:-../pdft_checkpointnewb6.pkl}"
CHECKREF="${2:-al2.4_sigma0.002_last_dm.pkl}"
XYZ="${3:-3.xyz}"
GRID_LEVEL="${4:-3}"
OLD_CHK="${5:-../al2.4.chk}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "PWD: $(pwd)"
echo "checkpoint=${CHECKPOINT}  checkref=${CHECKREF}"
echo "xyz=${XYZ}  grid_level=${GRID_LEVEL}  old_chk=${OLD_CHK}"
echo

python -u compare_density_dlnew_drnew_nf.py \
  --checkpoint "${CHECKPOINT}" \
  --checkref   "${CHECKREF}" \
  --xyz        "${XYZ}" \
  --grid-level "${GRID_LEVEL}" \
  --old-chk    "${OLD_CHK}"

echo
echo "Done."
