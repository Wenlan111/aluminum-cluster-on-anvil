#!/bin/bash
#SBATCH -A che240225
#SBATCH -J alclusterpyscf
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=128G                      # or set an explicit value, e.g. 64G
#SBATCH --constraint=A
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --output=%x-%J.out
#SBATCH --error=%x-%J.err
#SBATCH --hint=nomultithread         # optional; prefer physical cores

set -euo pipefail

# -------- Env ----------
module purge
module load monitor
module load conda
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate n2v_envi
set -u
# Define MKL vars safely (works even with nounset)
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-LP64}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-INTEL}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
/usr/bin/time -v srun --cpu-bind=cores python -u alcluster.py 
export TMPDIR=${TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
mkdir -p "$TMPDIR"

# Let PySCF know its memory budget (MB), keep ~2GB headroom
if [[ -r /proc/meminfo ]]; then
  export PYSCF_MAX_MEMORY=$(( ($(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1024) - 2048 ))
fi

echo "=== Job info ==="
echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID  Node: $(hostname)"
echo "Threads: $OMP_NUM_THREADS  PySCF max mem (MB): ${PYSCF_MAX_MEMORY:-unset}"
lscpu | egrep 'Model name|Socket|Thread|Core|NUMA|CPU\(s\)' || true
free -h || true
echo "TMPDIR=$TMPDIR"

echo "All done."

