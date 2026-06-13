# 2.4 Å inversion and NAD energies

Self-contained directory for n2v inversion and non-additive (NAD) energy
decomposition at **d = 2.4 Å**, matching the layout of `2.7inversion/`.

## Run

```bash
cd 2.4inversion
sbatch job_inversion.sh    # n2v inversion → inversion_ks_for_energy3.pkl
sbatch Excnad.sh           # NAD terms via calculateexc.py
```

## Inputs (paths in scripts point to `../2.4pdft` and `../2.4dftref`)

| File | Role |
|------|------|
| `../2.4pdft/pdft_checkpointnewref5.pkl` | Latest PDFT fragment DMs (`Dal`…`Dbr`) |
| `../2.4dftref/al2.4_sigma0.002.chk` | Full-system `mol` (geometry + AO basis) |
| `../2.4dftref/al2.4_sigma0.002_last_dm.pkl` | Reference α/β DMs for `nref` / `rho_ref` |
| `o.xyz`, `rgeo.xyz`, `onew.xyz` | Fragment / embedded geometries |
| `inversion_ks_for_energy3.pkl` | Wu–Yang warm-start (optional) |

Symlinks with the same names may exist in this directory for convenience.

## Outputs (written here)

- `inversion_ks_for_energy3.pkl` — converged inversion KS densities
- `inversion_v_pbs_last_iter.pkl` — restart on failure
- Slurm `inversion*` / `Excnad*` logs
