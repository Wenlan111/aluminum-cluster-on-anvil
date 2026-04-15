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

#read the pkl file
with open("pdft_checkpointnewb6.pkl", "rb") as f:
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
mol, scf_dict = scfchk.load_scf('al2.4.chk')
mf = dft.UKS(mol)
mo_coeff  = scf_dict['mo_coeff']
mo_occ    = scf_dict['mo_occ']
mo_energy = scf_dict['mo_energy']
Daref, Dbref = mf.make_rdm1(mo_coeff, mo_occ) 

#print the data
Dal = data["Dal"]
Dbl = data["Dbl"]
Dar = data["Dar"]
Dbr = data["Dbr"]

#reconstruct AO space for each fragment
w1o = 0.332211
w2o = 0.667789
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

w1al =0.6661055
w2al =0.3338945
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
inv.set_system(mol, "6-311++G(d,p)", pbs="6-31g*")
inv.Dt = Dtotnew

# Inverter with WuYang method, guide potention v0=Fermi-Amaldi
_t0 = time.perf_counter()
inv.invert("WuYang", guide_components="fermi_amaldi")
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
print("Ne_inv =", Ne_inv)
T_ao = inv.T
T_kinetic = float(np.einsum("ij,ji", Da, T_ao) + np.einsum("ij,ji", Db, T_ao))
print("T (electronic kinetic, Ha) =", T_kinetic)

V_ao = inv.V
V_nuclear = float(np.einsum("ij,ji", Da, V_ao) + np.einsum("ij,ji", Db, V_ao))
print("V (nuclear attraction, Ha) =", V_nuclear)


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
        },
        f,
        protocol=pickle.HIGHEST_PROTOCOL,
    )


