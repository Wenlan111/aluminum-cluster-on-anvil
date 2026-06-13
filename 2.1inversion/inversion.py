import os
import sys
import pyscf
from pyscf import gto
from pyscf import dft
import numpy as np
import pickle
import time
from pyscf.scf import chkfile as scfchk

# n2v must be patched before fragments / anything else that could touch n2v.
# Stock PySCFGrider.__init__ calls mf.kernel() and allocates dense int2e (OOM for large systems).
import n2v
import n2v.grid.pyscfgrider as _n2v_pgmod
from gbasis.wrappers import from_pyscf
import n2v.engines.pyscf as _n2v_pyscf_eng
import n2v.inverter as _n2v_inverter


def _grider_init_no_scf(self, mol, pbs_mol):
    self.mol = mol
    self.basis = from_pyscf(mol)
    self.pbs = from_pyscf(pbs_mol) if pbs_mol is not None else None
    self.atomic_charges = self.mol.atom_charges()
    self.atomic_coords = self.mol.atom_coords()
    mf = dft.UKS(self.mol)
    mf.xc = "svwn"
    mf.grids = dft.gen_grid.Grids(self.mol)
    mf.grids.level = 1
    mf.grids.build()
    self.spherical_points = mf.grids.coords
    self.w = mf.grids.weights
    self.mf = mf
    self.rectangular_grid = None


_n2v_pgmod.PySCFGrider.__init__ = _grider_init_no_scf
# Same class object as engines will import from n2v.grid
if "n2v.grid" in sys.modules:
    _ng = sys.modules["n2v.grid"]
    if hasattr(_ng, "PySCFGrider"):
        assert _ng.PySCFGrider is _n2v_pgmod.PySCFGrider


def _compute_hartree_df(self, Cocc_a, Cocc_b=None):
    da = Cocc_a @ Cocc_a.T
    if Cocc_b is not None:
        db = Cocc_b @ Cocc_b.T
    else:
        db = da
    mf = dft.UKS(self.mol).density_fit(auxbasis="weigend")
    return mf.get_j(dm=[da, db])


_n2v_pyscf_eng.PySCFEngine.compute_hartree = _compute_hartree_df


def _generate_components_from_dm(self, guide_components, **keywords):
    """Build guide potentials from Dt only; no ct/CI required (same as inversionZMP.py)."""
    self.guide_components = guide_components
    self.va = np.zeros((self.nbf, self.nbf))
    self.vb = np.zeros((self.nbf, self.nbf))
    n_tot = self.nalpha + self.nbeta

    da_t = self.Dt[0]
    db_t = self.Dt[1] if self.ref == 2 else self.Dt[0]
    mfj = dft.UKS(self.eng.mol).density_fit(auxbasis="weigend")
    ja, jb = mfj.get_j(dm=[da_t, db_t])
    self.J0 = [ja, jb]

    if guide_components == "none":
        return
    if guide_components == "hartree":
        v_h = self.J0[0] + self.J0[1]
        self.va += v_h
        self.vb += v_h
        return
    if guide_components == "fermi_amaldi":
        v_fa = (1 - 1 / n_tot) * (self.J0[0] + self.J0[1])
        self.va += v_fa
        self.vb += v_fa
        return
    raise ValueError("Guide component not recognized")


_n2v_inverter.Inverter.generate_components = _generate_components_from_dm

# Stock get_S3 uses the full grid at once; accumulate in chunks for large systems.
_S3_GRID_CHUNK = 4096


def _get_S3_chunked(self):
    grid = dft.gen_grid.Grids(self.mol)
    grid.level = 1
    grid.build()
    npts = int(grid.weights.size)
    nao = self.mol.nao_nr()
    if self.pbs_str == "same":
        out = np.zeros((nao, nao, nao))
        for start in range(0, npts, _S3_GRID_CHUNK):
            end = min(start + _S3_GRID_CHUNK, npts)
            coords = grid.coords[start:end]
            w = grid.weights[start:end]
            bs = dft.numint.eval_ao(self.mol, coords)
            out += np.einsum("g,gj,gk,gl->jkl", w, bs, bs, bs, optimize=True)
    else:
        npbs = self.npbs
        out = np.zeros((nao, nao, npbs))
        for start in range(0, npts, _S3_GRID_CHUNK):
            end = min(start + _S3_GRID_CHUNK, npts)
            coords = grid.coords[start:end]
            w = grid.weights[start:end]
            bs1 = dft.numint.eval_ao(self.mol, coords)
            bs2 = dft.numint.eval_ao(self.pbs, coords)
            out += np.einsum("g,gj,gk,gl->jkl", w, bs1, bs1, bs2, optimize=True)
    return out


_n2v_pyscf_eng.PySCFEngine.get_S3 = _get_S3_chunked

# Wrap get_S3: optional DF path (import after chunked assignment so fallback stays chunked).
S3_USE_DENSITY_FIT = os.environ.get("INVERSION_S3_DF", "1").lower() not in (
    "0",
    "false",
    "no",
    "",
)
import n2v_pyscf_engine_df_option  # noqa: E402

if _n2v_pgmod.PySCFGrider.__init__ is not _grider_init_no_scf:
    raise RuntimeError("n2v PySCFGrider patch did not apply (wrong n2v install?).")
print("n2v PySCFGrider memory patch: active", flush=True)
print(
    "n2v get_S3: chunked (chunk=%d); DF wrapper (INVERSION_S3_DF=%s)"
    % (_S3_GRID_CHUNK, S3_USE_DENSITY_FIT),
    flush=True,
)
print("n2v from:", getattr(n2v, "__file__", "?"), flush=True)

import fragments

# Paths (match 2.4pdft/2.4capdft.py: latest PDFT + smeared d=2.4 dftref, sigma=0.002)
_DIR = os.path.dirname(os.path.abspath(__file__))
PDFT_CHECKPOINT = os.path.join(_DIR, "..", "2.1pdft", "pdft2.1_checkpointnewref5.pkl")
DFTREF_CHK = os.path.join(_DIR, "..", "2.1dftref", "al2.1_sigma0.002.chk")
DFTREF_DM_PKL = os.path.join(_DIR, "..", "2.1dftref", "al2.1_sigma0.002_last_dm.pkl")

with open(PDFT_CHECKPOINT, "rb") as f:
    data = pickle.load(f)

#define the target AO space
obasis = gto.basis.parse('''O    S
   8588.500                  0.00189515
   1297.230                  0.0143859
    299.2960                 0.0707320
     87.37710                0.2400010
     25.67890                0.5947970
      3.740040               0.2808020
O    SP
     42.11750                0.113889               0.0365114
      9.628370               0.920811               0.237153
      2.853320              -0.00327447             0.819702
O    SP
      0.905661               1.000000               1.000000
O    SP
      0.255611               1.000000               1.000000
O    SP
      0.0845000              1.0000000              1.0000000
O    D
      1.292                  1.000000''')

albasis = gto.basis.parse('''Al    S
  54866.489                  0.000839
   8211.7665                 0.006527
   1866.1761                 0.033666
    531.12934                0.132902
    175.11797                0.401266
     64.005500               0.531338
Al    S
     64.005500               0.202305
     25.292507               0.624790
     10.534910               0.227439
Al    S
      3.2067110              1.000000
Al    S
      1.152555               1.000000
Al    S
      0.1766780              1.000000
Al    S
      0.0652370              1.000000
Al    P
    259.28362                0.009448
     61.076870               0.070974
     19.303237               0.295636
      7.0108820              0.728219
Al    P
      2.6738650              0.644467
      1.0365960              0.417413
Al    P
      0.3168190              1.000000
Al    P
      0.1142570              1.000000
Al    SP
      0.0318000              1.0000000              1.0000000
Al    P
      0.041397               1.000000
Al    D
      0.3250000              1.0000000''')
def _resolve_mol_xyz(mol, chk_path):
    """Chk files store mol.atom as a bare filename (e.g. 3.xyz); resolve via dftref dir."""
    atom = mol.atom
    if isinstance(atom, str) and atom.lower().endswith(".xyz") and not os.path.isfile(atom):
        candidate = os.path.join(os.path.dirname(chk_path), os.path.basename(atom))
        if os.path.isfile(candidate):
            mol.atom = os.path.abspath(candidate)
    return mol


# Full-system mol (geometry + AO basis) from dftref chk — required for inv.set_system.
mol, _scf_dict = scfchk.load_scf(DFTREF_CHK)
mol = _resolve_mol_xyz(mol, DFTREF_CHK)
with open(DFTREF_DM_PKL, "rb") as f:
    _dftref = pickle.load(f)
Daref = _dftref["Da_last_normal"]
Dbref = _dftref["Db_last_normal"]

#print the data
Dal = data["Dal"]
Dbl = data["Dbl"]
Dar = data["Dar"]
Dbr = data["Dbr"]

#reconstruct AO space for each fragment
w1o = 0.280274
w2o = 0.719726
lgeo1=gto.Mole()
lgeo1.atom= "o.xyz"
lgeo1.unit='angstrom'
lgeo1.basis = {'O': obasis, 'ghost-Al': '6-31g*'}

#lgeo1.basis = 'def2-SVP'
lgeo1.spin = 2
lgeo1.build()
ldft1 = fragments.FragmentDFT(lgeo1,'pbe', newton = True)
lgeo2=gto.Mole()
lgeo2.atom="o.xyz"
lgeo2.unit='angstrom'
lgeo2.basis = {'O': obasis,'ghost-Al':'6-31g*'}
#lgeo2.basis = 'def2-SVP'
lgeo2.charge = -1
lgeo2.spin = 1
lgeo2.build()
ldft2 = fragments.FragmentDFT(lgeo2,'pbe', newton = True)
l = fragments.ens([ldft1,ldft2],[w1o,w2o])

w1al = 0.640137
w2al = 0.359863
rgeo1 = gto.Mole()
rgeo1.atom = "rgeo.xyz"
rgeo1.unit = 'angstrom'
rgeo1.basis = {'ghost-O': '6-31g*', 'Al': albasis}
rgeo1.spin = 0
rgeo1.build()
rdft1 = fragments.FragmentDFT(rgeo1,'pbe',metal = True, smearing = True, newton =False)
#rdft1.dftsolver = remove_linear_dep_(rdft1.dftsolver, lindep=1e-4)
#rgeo2
rgeo2 = gto.Mole()
rgeo2.atom = "rgeo.xyz"
rgeo2.unit = 'angstrom'
rgeo2.basis = {'ghost-O': '6-31g*', 'Al': albasis}
rgeo2.charge = 2
rgeo2.spin = 2
rgeo2.build()
rdft2 = fragments.FragmentDFT(rgeo2,'pbe',metal = True,smearing =True, newton= False,sigma=0.005)
r = fragments.ens([rdft1,rdft2],[w1al,w2al])


#new way: embed the dm to the full space
rgeonew = gto.Mole()
rgeonew.atom = "rgeo.xyz"
rgeonew.unit = 'angstrom'
rgeonew.basis = {'ghost-O': obasis, 'Al': albasis}
rgeonew.build()


lgeonew = gto.Mole()
lgeonew.atom = "onew.xyz"
lgeonew.unit = 'angstrom'
lgeonew.basis = {'O': obasis, 'ghost-Al': albasis}
lgeonew.build()

Sl = lgeonew.intor('int1e_ovlp')
Tl = gto.intor_cross('int1e_ovlp', lgeo1,lgeonew)
Sr = rgeonew.intor('int1e_ovlp')
Tr = gto.intor_cross('int1e_ovlp', rgeo1,rgeonew)
def project_one(D_old,S,T):
    M = T.T @ D_old @ T
    X = np.linalg.solve(S, M)
    return np.linalg.solve(S, X.T).T

Dalnew = project_one(Dal,Sl,Tl)
Dblnew = project_one(Dbl,Sl,Tl)
Darnew = project_one(Dar,Sr,Tr)
Dbrnew = project_one(Dbr,Sr,Tr)


Dtotnew = (Dalnew + Darnew, Dblnew + Dbrnew)

#inverter

# Initialize inverter object.
inv = n2v.Inverter(engine="pyscf")
inv.eng.s3_use_density_fit = S3_USE_DENSITY_FIT
if S3_USE_DENSITY_FIT:
    inv.eng.s3_df_auxbasis = os.environ.get("INVERSION_S3_DF_AUXBASIS", "weigend")
_inv_basis = {"O": obasis, "Al": albasis}
# ref=1 (RKS-style): inverter fills 2*nalpha orbitals with one shared v_pbs.
# This collapses the open-shell asymmetry into a closed-shell-like fit and
# gives a much better-conditioned WY problem. ninv = 2*nalpha = 668 will
# differ from the smeared-PDFT target's 666; this is a known projection
# artifact of ref=1 and is tolerated here in exchange for stable convergence.
inv.set_system(mol, _inv_basis, pbs="6-311g**", ref=2)
inv.Dt = Dtotnew
S2_inv = inv.S2
inv.nalpha = int(np.rint(np.einsum("ij,ji", inv.Dt[0], S2_inv)))
inv.nbeta = int(np.rint(np.einsum("ij,ji", inv.Dt[1], S2_inv)))

# Wu–Yang initial guess: load v_pbs from a prior inversion pickle (same basis/PBS).
# For ref=2 UKS, the pickle must store the full vector, length 2*npbs (no [v,v] join).
# After a failed run this script writes inversion_v_pbs_last_iter.pkl — prefer that on retry.
_last_vp_path = "inversion_v_pbs_last_iter.pkl"
_prior_vp_path = "inversion_ks_for_energy3.pkl"
if os.path.isfile(_last_vp_path):
    V_PBS_WARMSTART_PKL = _last_vp_path
elif os.path.isfile(_prior_vp_path):
    V_PBS_WARMSTART_PKL = _prior_vp_path
else:
    V_PBS_WARMSTART_PKL = None
if V_PBS_WARMSTART_PKL and os.path.isfile(V_PBS_WARMSTART_PKL):
    with open(V_PBS_WARMSTART_PKL, "rb") as _f:
        _warm = pickle.load(_f)
    v0 = np.asarray(_warm["v_pbs"], dtype=np.float64).ravel()
    need = inv.v_pbs.shape
    _saved_ref = _warm.get("ref")
    if v0.shape == need:
        inv.v_pbs = v0.copy()
        print(
            f"Wu–Yang x0 from {V_PBS_WARMSTART_PKL}: v_pbs shape {inv.v_pbs.shape}",
            flush=True,
        )
    elif (
        _saved_ref == 1
        and inv.ref == 2
        and v0.size == inv.npbs
        and need == (2 * inv.npbs,)
    ):
        inv.v_pbs = np.concatenate([v0, v0]).copy()
        print(
            f"Wu–Yang x0 from {V_PBS_WARMSTART_PKL}: expanded ref=1 v_pbs → {inv.v_pbs.shape}",
            flush=True,
        )
    else:
        print(
            f"[warm-start skipped] {V_PBS_WARMSTART_PKL}: len(v_pbs)={v0.size}, need {need[0]} "
            f"(run npbs={inv.npbs}, pickle ref={_saved_ref!r}).",
            flush=True,
        )
        print(
            "  → Length mismatch: pickle is not from this set_system (PBS/basis/ref differ). "
            "Use V_PBS_WARMSTART_PKL = None, or a pickle from a completed inversion with the same mol/basis/PBS.",
            flush=True,
        )

_t0 = time.perf_counter()
try:
    inv.invert(
        "WuYang",
        guide_components="hartree",
        gtol=1e-2,
        opt_max_iter=2000,
        opt_method="bfgs",
    )
except Exception:
    # Full results pickle is only written after a successful run. On failure, n2v still
    # leaves inv.v_pbs at the last iterate — save it for the next warm-start attempt.
    _last_vp = "inversion_v_pbs_last_iter.pkl"
    try:
        with open(_last_vp, "wb") as _sf:
            pickle.dump(
                {
                    "v_pbs": np.asarray(inv.v_pbs).copy(),
                    "ref": int(getattr(inv, "ref", 1)),
                    "npbs": int(getattr(inv, "npbs", 0)),
                    "ok": False,
                },
                _sf,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(
            f"Wu–Yang did not converge; saved last v_pbs ({inv.v_pbs.shape}) to {_last_vp}",
            flush=True,
        )
    except Exception as _save_e:
        print(f"Could not save last v_pbs: {_save_e}", flush=True)
    raise
_invert_s = time.perf_counter() - _t0
print(f"Inversion (WuYang) wall time: {_invert_s:.3f} s", flush=True)
Da = inv.Da
Db = inv.Db
Ca = inv.Ca
Cb = inv.Cb
Coca = inv.Coca
Cocb = inv.Cocb
S = inv.S2
Ne_inv = np.einsum("ij,ji", Da + Db, S)
print("nalpha =", inv.nalpha)
print("nbeta =", inv.nbeta)
Ne_a = np.einsum("ij,ji", Da, S)
Ne_b = np.einsum("ij,ji", Db, S)
print("Ne_a =", Ne_a)
print("Ne_b =", Ne_b)
print("Ne_inv =", Ne_inv)
T_ao = inv.T
T_kinetic = float(np.einsum("ij,ji", Da, T_ao) + np.einsum("ij,ji", Db, T_ao))
print("T (electronic kinetic, Ha) =", T_kinetic)

V_ao = inv.V
V_nuclear = float(np.einsum("ij,ji", Da, V_ao) + np.einsum("ij,ji", Db, V_ao))
print("V (nuclear attraction, Ha) =", V_nuclear)

# Hartree from inverted density (DF Coulomb for memory safety).
# E_H = (1/2) Tr[(Da+Db) . J[Da+Db]]  (includes αβ cross terms).
mf_j = dft.UKS(mol).density_fit(auxbasis="weigend")
_D_tot = Da + Db
_J_tot = mf_j.get_j(dm=_D_tot)
E_hartree = 0.5 * float(np.einsum("ij,ji", _D_tot, _J_tot))
print("E_hartree (Ha) =", E_hartree)

# XC from inverted density on numerical grid.
XC_CODE = "PBE"
mf_xc = dft.UKS(mol)
mf_xc.xc = XC_CODE
mf_xc.grids.level = 3
mf_xc.grids.build()
ao_xc = dft.numint.eval_ao(mol, mf_xc.grids.coords, deriv=1)
rho_a = dft.numint.eval_rho(mol, ao_xc, Da, xctype="GGA")
rho_b = dft.numint.eval_rho(mol, ao_xc, Db, xctype="GGA")
exc_eps = dft.libxc.eval_xc(XC_CODE, (rho_a, rho_b), spin=1)[0]
rho_tot = rho_a[0] + rho_b[0]
E_xc = float(np.dot(mf_xc.grids.weights, exc_eps * rho_tot))
print(f"E_xc ({XC_CODE}, Ha) =", E_xc)

E_nn = float(mol.energy_nuc())
print("E_nn (nuclear repulsion, Ha) =", E_nn)

E_ks_total = T_kinetic + V_nuclear + E_hartree + E_xc + E_nn
print("E_ks_total (Ha) =", E_ks_total)


# Compare electron counts: nref (dftref last-normal DM), nf (PDFT projected),
# ninv (inverted density).


S_full = mol.intor("int1e_ovlp")
nref = float(np.einsum("ij,ji", Daref + Dbref, S_full))
nf = float(np.einsum("ij,ji", Dtotnew[0] + Dtotnew[1], S_full))
ninv = float(Ne_inv)

# nf2: electrons from the raw pkl fragment densities in their own AO spaces
# (left in lgeo1, right in rgeo1), summed over the two fragments.
S_l_frag = lgeo1.intor("int1e_ovlp")
S_r_frag = rgeo1.intor("int1e_ovlp")
nf2_l = float(np.einsum("ij,ji", Dal + Dbl, S_l_frag))
nf2_r = float(np.einsum("ij,ji", Dar + Dbr, S_r_frag))
nf2 = nf2_l + nf2_r

print("=" * 60)
print("Electron-count comparison")
print("-" * 60)
print(f"nref (dftref sigma=0.002 DM)  = {nref:.6f}")
print(f"nf   (pkl projected to mol)   = {nf:.6f}")
print(f"nf2  (pkl in fragment AOs)    = {nf2:.6f}  (L={nf2_l:.6f}, R={nf2_r:.6f})")
print(f"ninv (inverted density)       = {ninv:.6f}")
print("-" * 60)
print(f"nf   - nref = {nf - nref:+.3e}")
print(f"nf2  - nref = {nf2 - nref:+.3e}")
print(f"ninv - nref = {ninv - nref:+.3e}")
print(f"nf   - nf2  = {nf - nf2:+.3e}")
print(f"ninv - nf   = {ninv - nf:+.3e}")
print("=" * 60, flush=True)

# --- Density (real-space) comparison on a common grid -----------------------
# Build a grid on mol and evaluate each density rho(r) = rho_a + rho_b.
_grid = dft.gen_grid.Grids(mol)
_grid.level = 3
_grid.build()
_coords = _grid.coords
_w = _grid.weights

# AO values of mol, lgeo1, rgeo1 on the SAME grid points.
_ao_mol = dft.numint.eval_ao(mol, _coords)
_ao_l = dft.numint.eval_ao(lgeo1, _coords)
_ao_r = dft.numint.eval_ao(rgeo1, _coords)

def _rho_from_dm(ao, Da_, Db_):
    rho_a = dft.numint.eval_rho(mol, ao, Da_, xctype="LDA")
    rho_b = dft.numint.eval_rho(mol, ao, Db_, xctype="LDA")
    return rho_a + rho_b

rho_ref = _rho_from_dm(_ao_mol, Daref, Dbref)
rho_f = _rho_from_dm(_ao_mol, Dtotnew[0], Dtotnew[1])
rho_inv = _rho_from_dm(_ao_mol, Da, Db)

# rho_f2: evaluate the raw pkl fragment densities in their own AO spaces on the
# same grid, then add (left in lgeo1 AOs, right in rgeo1 AOs).
rho_f2_l = dft.numint.eval_rho(lgeo1, _ao_l, Dal, xctype="LDA") + dft.numint.eval_rho(
    lgeo1, _ao_l, Dbl, xctype="LDA"
)
rho_f2_r = dft.numint.eval_rho(rgeo1, _ao_r, Dar, xctype="LDA") + dft.numint.eval_rho(
    rgeo1, _ao_r, Dbr, xctype="LDA"
)
rho_f2 = rho_f2_l + rho_f2_r

def _int(x):
    return float(np.dot(_w, x))

def _cmp(label, rA, rB):
    l1 = _int(np.abs(rA - rB))
    linf = float(np.max(np.abs(rA - rB)))
    print(f"{label:<22s}  int|dρ|={l1:.4e}  max|dρ|={linf:.4e}")

print("=" * 60)
print("Density (real-space) comparison on mol grid (level=3)")
print("-" * 60)
print(f"int rho_ref = {_int(rho_ref):.6f}")
print(f"int rho_f   = {_int(rho_f):.6f}")
print(f"int rho_f2  = {_int(rho_f2):.6f}  (L={_int(rho_f2_l):.6f}, R={_int(rho_f2_r):.6f})")
print(f"int rho_inv = {_int(rho_inv):.6f}")
print("-" * 60)
_cmp("rho_f   - rho_ref", rho_f, rho_ref)
_cmp("rho_f2  - rho_ref", rho_f2, rho_ref)
_cmp("rho_inv - rho_ref", rho_inv, rho_ref)
_cmp("rho_f   - rho_f2 ", rho_f, rho_f2)
_cmp("rho_inv - rho_f  ", rho_inv, rho_f)
_cmp("rho_inv - rho_f2 ", rho_inv, rho_f2)
print("=" * 60, flush=True)

_energy_pkl = "inversion_ks_for_energy3.pkl"
with open(_energy_pkl, "wb") as f:
    pickle.dump(
        {
            "Da": Da,
            "Db": Db,
            "Ca": Ca,
            "Cb": Cb,
            "Coca": Coca,
            "Cocb": Cocb,
            "Dtotnew": Dtotnew,
            "Ne_inv": Ne_inv,
            "S": S,
            "T": T_ao,
            "V": V_ao,
            "T_kinetic": T_kinetic,
            "V_nuclear": V_nuclear,
            "E_hartree": E_hartree,
            "E_xc": E_xc,
            "E_nn": E_nn,
            "E_ks_total": E_ks_total,
            "xc_code": XC_CODE,
            "v_pbs": np.asarray(inv.v_pbs).copy(),
            "guide_components": "hartree",
            "gtol": 1e-3,
            "ref": int(getattr(inv, "ref", 1)),
            "npbs": int(getattr(inv, "npbs", inv.v_pbs.size)),
        },
        f,
        protocol=pickle.HIGHEST_PROTOCOL,
    )


