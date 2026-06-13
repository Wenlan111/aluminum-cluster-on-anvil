#!/bin/bash
#SBATCH -A che240225
#SBATCH -J inversion
#SBATCH -t 6:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -p highmem
#SBATCH --error=inversion%x-%j.err
#SBATCH --out=inversion%x-%j.out
#SBATCH --mail-user=miao74@purdue.edu
#SBATCH --mail-type=all

# Run from the directory that has inversion.py, *.xyz, *.pkl, *.chk:
#   sbatch job_inversion.sh
# Use shared instead of highmem if the job fits in shared memory and you want a shorter queue:
#   #SBATCH -p shared

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

SCRIPT=inversion.py

python -u "$SCRIPT"
echo "=== Python finished ==="
echo "All done."
