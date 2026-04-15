#!/bin/bash
#SBATCH -A che240225
#SBATCH -J extractexc_all
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -p shared
# Default RAM on shared is often small; extractexc uses (4×ngrid×nao) floats for GGA AO.
# Raise if you still see OOM (Slurm: "oom_kill" / exit 137 / "Killed").
#SBATCH --mem=128G
#SBATCH --error=extractexc_all-%j.err
#SBATCH --out=extractexc_all-%j.out

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

WORKDIR="/anvil/scratch/x-wmiao/alcluster50/Al_cluster/anew/newbasis/nadexc"
SCRIPT="${WORKDIR}/extractexc.py"
XC_CODE="${1:-PBE}"

cd "${WORKDIR}"
echo "Working directory: ${WORKDIR}"
echo "Using XC functional: ${XC_CODE}"
echo

shopt -s nullglob
chk_files=( *.chk )
if [ ${#chk_files[@]} -eq 0 ]; then
  echo "No .chk files found."
  exit 1
fi

for chk in "${chk_files[@]}"; do
  if [ ! -s "${chk}" ]; then
    echo "Skipping empty chk file: ${chk}"
    continue
  fi

  base="${chk%.chk}"
  out_file="${base}.exc.txt"

  echo "=== Processing ${chk} ==="
  python -u "${SCRIPT}" "${chk}" "${XC_CODE}" > "${out_file}" 2>&1
  echo "Saved: ${out_file}"
  echo
done

echo "All done."
