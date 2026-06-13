#!/bin/bash
#SBATCH -A che240225
#SBATCH -J pdftl1
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -p highmem
#SBATCH --error=l1opt2.4-%j.err
#SBATCH --out=l1opt2.4-%j.out
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

python -u get_l1_from_pkl.py
echo "=== Python finished ==="
echo "All done."
