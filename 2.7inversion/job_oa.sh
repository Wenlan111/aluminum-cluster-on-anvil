#!/bin/bash
#SBATCH -A che240225
#SBATCH -J oa
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -p highmem
#SBATCH --error=oa-%j.err
#SBATCH --out=oa-%j.out
#SBATCH --mail-user=miao74@purdue.edu
#SBATCH --mail-type=all


module purge
module load conda
module load monitor
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate n2v_envi
set -u

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export KMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

SCRIPT=oa.py

python -u "$SCRIPT"
echo "=== Python finished ==="
echo "All done."
