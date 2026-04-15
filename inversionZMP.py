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
import n2v.methods.zmp as _n2v_zmp


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
    """Build guide potentials from Dt only; no ct/CI required."""
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


def _zmp_generate_s_functional_from_dm(self, lam, Cocca, Coccb, Da, Db):
    """DM-driven ZMP penalty; ignores ct/Cocc inputs."""
    mfj = dft.UKS(self.eng.mol).density_fit(auxbasis="weigend")
    ja, jb = mfj.get_j(dm=[Da, Db if self.ref == 2 else Da])

    if self.ref == 1:
        vc_a = 2 * lam * (ja - self.J0[0])
        self.vca = ja - self.J0[0]
        return [vc_a]

    vc_a = lam * (ja - self.J0[0])
    vc_b = lam * (jb - self.J0[1])
    return [vc_a, vc_b]


_ORIG_ZMP_SCF = _n2v_zmp.ZMP.zmp_scf


def _zmp_scf_no_ct(self, lambda_list, maxiter, print_scf, D_conv):
    """ct fallback + per-λ / wall-time checkpoints (flush for Slurm logs)."""
    if not hasattr(self, "ct") or self.ct is None:
        self.ct = [None, None]

    if isinstance(lambda_list, np.ndarray):
        lam_arr = lambda_list.astype(float, copy=False).ravel()
    else:
        lam_arr = np.asarray(list(lambda_list), dtype=float).ravel()

    nlam = int(lam_arr.size)
    t0 = time.perf_counter()
    print(
        f"[ZMP] zmp_scf: nλ={nlam} maxiter={maxiter} D_conv={D_conv} "
        f"print_scf→True (SCF line every 5 iters)",
        flush=True,
    )

    def _lam_blocks():
        for i, v in enumerate(lam_arr):
            print(f"[ZMP] --- λ {i + 1}/{nlam}: λ={float(v):g} ---", flush=True)
            yield float(v)

    _ORIG_ZMP_SCF(self, _lam_blocks(), maxiter, True, D_conv)
    print(f"[ZMP] zmp_scf done in {time.perf_counter() - t0:.1f} s wall", flush=True)


def _zmp_with_checkpoints(
    self,
    opt_max_iter=100,
    opt_tol=None,
    lambda_list=None,
    zmp_mixing=1,
    print_scf=False,
):
    import psi4

    if opt_tol is None:
        opt_tol = psi4.core.get_option("SCF", "D_CONVERGENCE")
    if lambda_list is None:
        lambda_list = [70]
    self.diis_space = 100
    self.mixing = zmp_mixing
    lam = np.asarray(lambda_list).ravel()
    print("\nRunning ZMP:", flush=True)
    print(
        f"[ZMP] opt_max_iter={opt_max_iter} zmp_mixing={zmp_mixing} nλ={lam.size} "
        f"print_scf={print_scf} D_conv={opt_tol}",
        flush=True,
    )
    self.zmp_scf(lambda_list, opt_max_iter, print_scf, D_conv=opt_tol)


_n2v_zmp.ZMP.generate_s_functional = _zmp_generate_s_functional_from_dm
_n2v_zmp.ZMP.zmp_scf = _zmp_scf_no_ct
_n2v_zmp.ZMP.zmp = _zmp_with_checkpoints


def _set_basis_matrices_no_s3(self):
    """ZMP-only setup: skip S3 initialization."""
    self.T = self.eng.get_T()
    self.V = self.eng.get_V()
    self.A = self.eng.get_A()
    self.S2 = self.eng.get_S()
    self.S3 = None
    if self.eng.pbs_str != "same":
        self.T_pbs = self.eng.get_Tpbas()
    self.S4 = None


_n2v_inverter.Inverter.set_basis_matrices = _set_basis_matrices_no_s3

if _n2v_pgmod.PySCFGrider.__init__ is not _grider_init_no_scf:
    raise RuntimeError("n2v PySCFGrider patch did not apply (wrong n2v install?).")

import fragments

#read the pkl file
with open("pdft_checkpointnewb5.pkl", "rb") as f:
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


def _symmetrize_dm(D):
    return 0.5 * (D + D.T)

Dalnew = project_one(Dal,Sl,Tl)
Dblnew = project_one(Dbl,Sl,Tl)
Darnew = project_one(Dar,Sr,Tr)
Dbrnew = project_one(Dbr,Sr,Tr)


Dtotnew = (
    _symmetrize_dm(Dalnew + Darnew),
    _symmetrize_dm(Dblnew + Dbrnew),
)

#inverter

# Initialize inverter object.
inv = n2v.Inverter(engine="pyscf")
print("[ZMP] set_system (mol + pbs)...", flush=True)
_t_sys = time.perf_counter()
inv.set_system(mol, "6-311++G(d,p)")
print(f"[ZMP] set_system done in {time.perf_counter() - _t_sys:.1f} s", flush=True)
inv.Dt = Dtotnew
print("[ZMP] target Dt assigned; starting invert(zmp)...", flush=True)

# ZMP: Fermi–Amaldi guide; checkpoints from patched ZMP.zmp / zmp_scf
_t0 = time.perf_counter()
inv.invert(
    "zmp",
    opt_max_iter=200,
    opt_tol=1e-7,
    zmp_mixing=0.05,
    lambda_list=np.linspace(10, 1000, 20),
    guide_components="fermi_amaldi",
    print_scf=True,
)
_invert_s = time.perf_counter() - _t0
print(f"Inversion (ZMP) wall time: {_invert_s:.3f} s", flush=True)
Da = inv.Da
Db = inv.Db
Ca = inv.Ca
Cb = inv.Cb

_energy_pkl = "inversion_ks_for_energy.pkl"
with open(_energy_pkl, "wb") as f:
    pickle.dump(
        {
            "Da": Da,
            "Db": Db,
            "Ca": Ca,
            "Cb": Cb,
            "Dtotnew": Dtotnew,
        },
        f,
        protocol=pickle.HIGHEST_PROTOCOL,
    )


