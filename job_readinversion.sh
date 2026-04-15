#!/bin/bash
#SBATCH -A che240225
#SBATCH -J readinversion
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -p shared
#SBATCH --mem=8G
# Run where al2.4.chk and inversion_ks_for_energy.pkl live (edit if needed).
#SBATCH --chdir=/anvil/scratch/x-wmiao/alcluster50/Al_cluster/anew/newbasis
#SBATCH --error=readinversion-%j.err
#SBATCH --out=readinversion-%j.out

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

echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "PWD: $(pwd)"
echo "Python: $(command -v python)"
python -c "import pyscf; print('pyscf OK')"
echo
python -u readinversion.py
echo
echo "Done."
