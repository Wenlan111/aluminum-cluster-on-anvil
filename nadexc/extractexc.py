import os
import sys
import numpy as np
from pyscf import dft
from pyscf.scf import chkfile as scfchk


def _is_unrestricted_mo_occ(mo_occ):
    arr = np.asarray(mo_occ)
    return isinstance(mo_occ, (tuple, list)) or arr.ndim == 2


def _symm(dm):
    return 0.5 * (dm + dm.T)


if len(sys.argv) < 2:
    print("Usage: python extractexc.py your.chk [XC]")
    sys.exit(1)

chkname = sys.argv[1]
xc_code = sys.argv[2] if len(sys.argv) > 2 else "PBE"

print(f"Reading chkfile: {chkname}")
mol, scf_dict = scfchk.load_scf(chkname)

print("Number of electrons:", mol.nelectron)
print("Number of AO:", mol.nao_nr())
print("SCF total energy in chk:", scf_dict.get("e_tot"))

mo_occ = scf_dict["mo_occ"]
mo_coeff = scf_dict["mo_coeff"]
mo_energy = scf_dict.get("mo_energy")

# IMPORTANT: no SCF kernel is called anywhere in this script.
mf = dft.UKS(mol)
mf.xc = xc_code
# Level 3 is accurate but large: GGA needs eval_ao(deriv=1) ~ 4×ngrid×nao×8 bytes.
# If Slurm OOM-kills the job, set e.g. EXTRACTEXC_GRID_LEVEL=2 or 1 before running.
mf.grids.level = int(os.environ.get("EXTRACTEXC_GRID_LEVEL", "3"))
mf.grids.build()
print(f"DFT grid level={mf.grids.level}, ngrid={mf.grids.weights.size}, nao={mol.nao_nr()}")
Daref, Dbref = mf.make_rdm1(mo_coeff, mo_occ)

# Coulomb J: use density fitting — full int2e for ~1500 AO would need terabytes of RAM.
mf_j = dft.UKS(mol).density_fit(auxbasis="weigend")

if _is_unrestricted_mo_occ(mo_occ):
    da = _symm(Daref)
    db = _symm(Dbref)
    dm = (da, db)
    nelec_num, exc, vxc_ao = mf._numint.nr_uks(mol, mf.grids, mf.xc, dm)

    # GGA (e.g. PBE) needs ∂φ/∂r: deriv=1 → ao shape (4, ngrid, nao)
    ao = dft.numint.eval_ao(mol, mf.grids.coords, deriv=1)
    rho_a = dft.numint.eval_rho(mol, ao, da, xctype="GGA")
    rho_b = dft.numint.eval_rho(mol, ao, db, xctype="GGA")
    exc_grid = dft.libxc.eval_xc(mf.xc, (rho_a, rho_b), spin=1)[:2][0]
    rho_tot = rho_a[0] + rho_b[0]

    ja, jb = mf_j.get_j(dm=[da, db])
    e_hartree = 0.5 * (
        float(np.einsum("ij,ji", da, ja)) + float(np.einsum("ij,ji", db, jb))
    )
else:
    # Restricted chk: total density for XC; spin blocks for UKS get_j.
    da = _symm(Daref)
    db = _symm(Dbref)
    dm_r = da + db
    nelec_num, exc, vxc_ao = mf._numint.nr_rks(mol, mf.grids, mf.xc, dm_r)

    ao = dft.numint.eval_ao(mol, mf.grids.coords, deriv=1)
    rho = dft.numint.eval_rho(mol, ao, dm_r, xctype="GGA")
    exc_grid = dft.libxc.eval_xc(mf.xc, rho, spin=0)[:2][0]
    rho_tot = rho[0]

    ja, jb = mf_j.get_j(dm=[da, db])
    e_hartree = 0.5 * (
        float(np.einsum("ij,ji", da, ja)) + float(np.einsum("ij,ji", db, jb))
    )

exc_total_from_grid = float(np.dot(mf.grids.weights, np.asarray(exc_grid) * np.asarray(rho_tot)))
print(f"XC functional: {mf.xc}")
print(f"Integrated electrons (numint): {nelec_num}")
print(f"E_xc from nr_*: {exc:.12f} Ha")
print(f"E_xc from grid integration: {exc_total_from_grid:.12f} Ha")
print(f"V_xc AO matrix shape: {np.asarray(vxc_ao).shape}")
print(f"E_hartree: {e_hartree:.12f} Ha")

# One-line sidecar next to the chk (Slurm also captures full stdout in *.exc.txt).
_base = os.path.splitext(os.path.basename(chkname))[0]
_chk_abs = os.path.abspath(chkname)
_hartree_path = os.path.join(os.path.dirname(_chk_abs), f"{_base}.hartree.txt")
with open(_hartree_path, "w", encoding="utf-8") as _hf:
    _hf.write(f"E_hartree_Ha {e_hartree:.12f}\n")
