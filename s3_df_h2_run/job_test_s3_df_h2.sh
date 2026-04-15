#!/bin/bash
#SBATCH -A che240225
#SBATCH -J test_s3_df_h2
#SBATCH -t 00:15:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -p shared
#SBATCH --mem=8G
# Parent newbasis (code); logs land in this submit directory
#SBATCH --chdir=/anvil/scratch/x-wmiao/alcluster50/Al_cluster/anew/newbasis/s3_df_h2_run
#SBATCH --error=test_s3_df_h2-%j.err
#SBATCH --out=test_s3_df_h2-%j.out

set -euo pipefail

NEWBASIS="/anvil/scratch/x-wmiao/alcluster50/Al_cluster/anew/newbasis"

module purge
module load conda
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate n2v_envi
set -u

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export KMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd "${NEWBASIS}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Log directory (Slurm -D): ${SLURM_SUBMIT_DIR:-.}"
echo "Code directory: ${NEWBASIS}"
echo "Python: $(command -v python)"
python -c "import pyscf; import n2v; print('pyscf OK, n2v OK')"
echo
python -u "${NEWBASIS}/test_s3_density_fit_h2.py"
echo
echo "Done."
