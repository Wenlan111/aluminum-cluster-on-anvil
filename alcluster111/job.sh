#!/bin/bash
#SBATCH -A che240225
#SBATCH -J alclusterpyscf
#SBATCH -t 40:00:00
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
echo

# -------- Live monitor (optional but handy) ----------
# Log average CPU and memory every 60s while job runs
monitor_interval=60
monitor_log="resource_monitor-$SLURM_JOB_ID.log"
monitor() {
  echo "Timestamp,JobID,AveCPU,MaxRSS,AveRSS,Tasks" >> "$monitor_log"
  while true; do
    ts=$(date +'%F %T')
    # Some Slurm setups want ".batch"
    sstat -j "${SLURM_JOB_ID}.batch" --format=JobID,AveCPU,MaxRSS,AveRSS,Tasks --noheader \
      | awk -v TS="$ts" '{gsub(/^ +| +$/,""); print TS","$0}' >> "$monitor_log" || true
    sleep "$monitor_interval"
  done
}
monitor & MON_PID=$!
# Ensure monitor stops even on errors
cleanup() { kill "$MON_PID" 2>/dev/null || true; }
trap cleanup EXIT

# -------- Run workload with detailed timing ----------
SCRIPT=alcluster.py   # <--- change to your script name

echo "=== Starting Python with /usr/bin/time -v ==="
# /usr/bin/time prints wall time, CPU%, and peak memory
/usr/bin/time -v srun --cpu-bind=cores python -u "$SCRIPT" 2>&1 | tee al_run-${SLURM_JOB_ID}.dat
echo "=== Python finished ==="

# Stop monitor and show last few samples
cleanup
echo
echo "=== Last resource samples (sstat) ==="
tail -n 10 "$monitor_log" || true

# -------- Final Slurm accounting summary ----------
echo
echo "=== sacct summary ==="
# First, a readable table:
sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,TotalCPU,AveCPU,NCPUS,MaxRSS,AveRSS,State,ExitCode
echo
# Also print a single-line, machine-parsable record (if you want to grep later):
sacct -j "$SLURM_JOB_ID" --parsable2 --noheader \
  --format=JobID,Elapsed,TotalCPU,AveCPU,NCPUS,MaxRSS,AveRSS,State,ExitCode

echo
echo "All done."

