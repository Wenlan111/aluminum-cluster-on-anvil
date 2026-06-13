#!/bin/bash
#SBATCH -A che240225
#SBATCH -J phase2sign2.7
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -p highmem
#SBATCH --error=phase2sign2.7-%j.err
#SBATCH --out=phase2sign2.7-%j.out
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

cd "/anvil/scratch/x-wmiao/alcluster50/Al_cluster/anew/newbasis/2.7pdft"

# Allowed values by script: 3 or 5
MAXITER="${MAXITER:-5}"

srun python -u phase2_only_sign_test_2_7.py --maxiter "${MAXITER}" --out-log phase2_only_sign_test_2.7.log

echo "=== phase2-only sign test finished ==="
