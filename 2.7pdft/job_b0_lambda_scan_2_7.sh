#!/bin/bash
#SBATCH -A che240225
#SBATCH -J b0scan2.7
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -p highmem
#SBATCH --error=b0scan2.7-%j.err
#SBATCH --out=b0scan2.7-%j.out
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
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun python -u b0_lambda_scan.py
echo "=== Python finished ==="
echo "All done."
