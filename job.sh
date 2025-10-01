#!/bin/bash
#SBATCH -A che240225
#SBATCH -J alclusterpyscf
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -p shared
#SBATCH --error=refmol2.err
#SBATCH --out=refmol2.out
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
SCRIPT=al.py   # <--- change to your script name

/usr/bin/time -v srun -u python -u  "$SCRIPT" 
echo "=== Python finished ==="


echo
echo "All done."

