#!/bin/bash
#SBATCH -A che240225
#SBATCH -J pdftca2.7
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -p highmem
#SBATCH --error=pdft2.7camol%x-%j.err
#SBATCH --out=pdft2.7camols%x-%j.out
#SBATCH --mail-user=miao74@purdue.edu
#SBATCH --mail-type=all

# -------- Env ----------
module purge
module load conda
module load monitor
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate n2v_envi
set -u
# Define MKL vars safely (works even with nounset)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export KMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}






# -------- Run workload with detailed timing ----------
SCRIPT=2.7capdft.py   # <--- change to your script name

python -u "$SCRIPT"
echo "=== Python finished ==="


echo
echo "All done."

