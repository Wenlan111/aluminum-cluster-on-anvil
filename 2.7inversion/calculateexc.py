import os
import pyscf
from pyscf import gto
from pyscf import dft
import numpy as np
import pickle
import fragments
from pyscf.scf import chkfile as scfchk

_DIR = os.path.dirname(os.path.abspath(__file__))
PDFT_CHECKPOINT = os.path.join(_DIR, "..", "2.7pdft", "pdft2.7_checkpointnewref.pkl")
DFTREF_CHK = os.path.join(_DIR, "..", "2.7dftref", "al2.7_sigma0.002.chk")

with open(PDFT_CHECKPOINT, "rb") as f:
    data = pickle.load(f)

#print the data
Dal = data["Dal"]
Dbl = data["Dbl"]
Dar = data["Dar"]
Dbr = data["Dbr"]
#basis
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
def D2n(D, *phis):
    ngrid = len(phis)
    n = 0.
    for phi in phis:
        n += np.einsum("pu,pv,uv->p", phi, phi, D, optimize=True)
    n /= ngrid
    return n


def _symm(dm):
    dm = np.asarray(dm)
    return 0.5 * (dm + dm.T)


def _hartree_uks_df(m, da, db, auxbasis="weigend"):
    """Electronic Hartree energy E_H = (1/2) Tr[(Da+Db) . J[Da+Db]] via DF.

    This is the correct UKS Hartree, including the αβ cross terms that are
    missed by 0.5*(Tr[Da Ja] + Tr[Db Jb]).
    """
    da = _symm(da)
    db = _symm(db)
    d_tot = da + db
    mfj = dft.UKS(m).density_fit(auxbasis=auxbasis)
    j_tot = mfj.get_j(dm=d_tot)
    return 0.5 * float(np.einsum("ij,ji", d_tot, j_tot))


# Full-system mol for grids / integrals (same basis as dftref).
mol, _scf_dict = scfchk.load_scf(DFTREF_CHK)

#reconstruct AO space for each fragment
w1o = 0.794146
w2o = 0.205854
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

w1al =0.897073
w2al =0.102927
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

grid = dft.gen_grid.Grids(mol)
grid.level = 3
grid.build()
coords = grid.coords
w = grid.weights

# ---------------------------------------------------------------------------
ao_values = dft.numint.eval_ao(mol, coords, deriv=1)
phi, phi_x, phi_y, phi_z = ao_values

ao_values_r1 = dft.numint.eval_ao(rgeo1, coords, deriv=1)
phi_r1, phi_r1_x, phi_r1_y, phi_r1_z = ao_values_r1

ao_values_l1 = dft.numint.eval_ao(lgeo1, coords, deriv=1)

phi_l1, phi_l1_x, phi_l1_y, phi_l1_z = ao_values_l1  

# Spin-resolved fragment densities (each fragment is open-shell, so PBE must
# see (rho_a, rho_b) with spin=1, not the spinless total).
rho_r_a = dft.numint.eval_rho(rgeo1, ao_values_r1, Dar, xctype="GGA")
rho_r_b = dft.numint.eval_rho(rgeo1, ao_values_r1, Dbr, xctype="GGA")
rho_l_a = dft.numint.eval_rho(lgeo1, ao_values_l1, Dal, xctype="GGA")
rho_l_b = dft.numint.eval_rho(lgeo1, ao_values_l1, Dbl, xctype="GGA")
rho_r = rho_r_a + rho_r_b
rho_l = rho_l_a + rho_l_b
rho_tot_a = rho_r_a + rho_l_a
rho_tot_b = rho_r_b + rho_l_b
rho_tot = rho_tot_a + rho_tot_b

xc = "PBE"
exc = dft.libxc.eval_xc(xc, (rho_tot_a, rho_tot_b), spin=1)[0]
Exc = float(np.dot(w, exc * rho_tot[0]))
print("Exc =", Exc)
excl = dft.libxc.eval_xc(xc, (rho_l_a, rho_l_b), spin=1)[0]
excr = dft.libxc.eval_xc(xc, (rho_r_a, rho_r_b), spin=1)[0]
Excl = float(np.dot(w, excl * rho_l[0]))
Excr = float(np.dot(w, excr * rho_r[0]))
print("Excl =", Excl, "Excr =", Excr)

print("Excna =", Exc - Excl - Excr)

S = mol.intor("int1e_ovlp")
# Hartree (DF, 1/2 Tr DJ): fragments on their mols; tot = chk ref density on full mol.
print("Nl =", np.einsum("ij,ji", Dalnew, S)+np.einsum("ij,ji", Dblnew, S))
print("Nr =", np.einsum("ij,ji", Darnew, S)+np.einsum("ij,ji", Dbrnew, S))
print("N =", np.einsum("ij,ji", Dalnew+Darnew, S)+np.einsum("ij,ji", Dblnew+Dbrnew, S))
Ehartreel = _hartree_uks_df(lgeo1, Dal, Dbl)
Ehartreer = _hartree_uks_df(rgeo1, Dar, Dbr)
Ehartreel_proj = _hartree_uks_df(lgeonew, Dalnew, Dblnew)
Ehartreer_proj = _hartree_uks_df(rgeonew, Darnew, Dbrnew)
Ehartreetot = _hartree_uks_df(mol, Dalnew+Darnew, Dblnew+Dbrnew)
print("Ehartreel =", Ehartreel)
print("Ehartreer =", Ehartreer)
print("Ehartreel_proj =", Ehartreel_proj)
print("Ehartreer_proj =", Ehartreer_proj)
print("Ehartreetot =", Ehartreetot)
print("Ehartree nad =", Ehartreetot - Ehartreel - Ehartreer)
print("Ehartreetot_projmolnad =", Ehartreetot - Ehartreel_proj - Ehartreer_proj)

#kinetic energy
Tl_ao = lgeo1.intor_symmetric("int1e_kin")
Tr_ao = rgeo1.intor_symmetric("int1e_kin")
T_kineticl = float(np.einsum("ij,ji", Dal, Tl_ao) + np.einsum("ij,ji", Dbl, Tl_ao))
T_kineticr = float(np.einsum("ij,ji", Dar, Tr_ao) + np.einsum("ij,ji", Dbr, Tr_ao))
print("T_kineticl =", T_kineticl, "T_kineticr =", T_kineticr)
print("T_kinetic =", T_kineticl+T_kineticr)
# Kinetic energy Tr(D T) for densities projected onto the full mol AO basis (Dalnew/Dblnew,
# Darnew/Dbrnew), same frame as mol overlap used for N above.
T_mol = mol.intor_symmetric("int1e_kin")
T_l_proj = lgeonew.intor_symmetric("int1e_kin")
T_r_proj = rgeonew.intor_symmetric("int1e_kin")
T_kineticl_proj = float(np.einsum("ij,ji", Dalnew, T_l_proj) + np.einsum("ij,ji", Dblnew, T_l_proj))
T_kineticr_proj = float(np.einsum("ij,ji", Darnew, T_r_proj) + np.einsum("ij,ji", Dbrnew, T_r_proj))
T_kinetic_projmolr = float(np.einsum("ij,ji", Darnew, T_r_proj) + np.einsum("ij,ji", Dbrnew, T_r_proj))
T_kinetic_projmoll = float(np.einsum("ij,ji", Dalnew, T_l_proj) + np.einsum("ij,ji", Dblnew, T_l_proj))
T_kinetic_projmol = float(np.einsum("ij,ji", Dalnew+Darnew, T_mol) + np.einsum("ij,ji", Dblnew+Dbrnew, T_mol))
print(
    "T_kineticl (projected Dl) =",
    T_kineticl_proj,
    "T_kineticr (projected Dr) =",
    T_kineticr_proj,
)
print("T_kinetic (projected Dl+Dr) =", T_kineticl_proj + T_kineticr_proj)
print("T_kinetic (projected Dr in mol frame) =", T_kinetic_projmolr)
print("T_kinetic (projected Dl in mol frame) =", T_kinetic_projmoll)
print("T_kinetic (projected Dl+Dr in mol frame) =", T_kinetic_projmol)
#nuclear energy
V_ao = mol.intor_symmetric("int1e_nuc") 
V_l = lgeo1.intor_symmetric("int1e_nuc")
V_r = rgeo1.intor_symmetric("int1e_nuc")
V_l_proj = lgeonew.intor_symmetric("int1e_nuc")
V_r_proj = rgeonew.intor_symmetric("int1e_nuc")
V_nuclear = float(np.einsum("ij,ji", Dalnew+Darnew, V_ao) + np.einsum("ij,ji", Dblnew+Dbrnew, V_ao))
rVnuclear = float(np.einsum("ij,ji", Dar, V_r) + np.einsum("ij,ji", Dbr, V_r))
lVnuclear = float(np.einsum("ij,ji", Dal, V_l) + np.einsum("ij,ji", Dbl, V_l))
rVnuclear_proj = float(np.einsum("ij,ji", Darnew, V_r_proj) + np.einsum("ij,ji", Dbrnew, V_r_proj))
lVnuclear_proj = float(np.einsum("ij,ji", Dalnew, V_l_proj) + np.einsum("ij,ji", Dblnew, V_l_proj))
V_nuclear_proj = float(np.einsum("ij,ji", Darnew, V_r_proj) + np.einsum("ij,ji", Dbrnew, V_r_proj) + np.einsum("ij,ji", Dalnew, V_l_proj) + np.einsum("ij,ji", Dblnew, V_l_proj))
rV_nuclear_projmol = float(np.einsum("ij,ji", Darnew, V_ao) + np.einsum("ij,ji", Dbrnew, V_ao))
lV_nuclear_projmol = float(np.einsum("ij,ji", Dalnew, V_ao) + np.einsum("ij,ji", Dblnew, V_ao))
V_nuclear_projmol = float(np.einsum("ij,ji", Darnew, V_ao) + np.einsum("ij,ji", Dbrnew, V_ao) + np.einsum("ij,ji", Dalnew, V_ao) + np.einsum("ij,ji", Dblnew, V_ao))
print("rVnuclear =", rVnuclear, "lVnuclear =", lVnuclear)
print("V_nuclear =", V_nuclear)
print("rVnuclear_proj =", rVnuclear_proj, "lVnuclear_proj =", lVnuclear_proj)
print("V_nuclear_proj =", V_nuclear_proj)
print("Vnuclearnad =", V_nuclear - rVnuclear - lVnuclear)
print("rV_nuclear_projmol =", rV_nuclear_projmol, "lV_nuclear_projmol =", lV_nuclear_projmol)
print("V_nuclear_projmol =", V_nuclear_projmol)
print("Vnuclearnad_projmol =", V_nuclear_projmol - rV_nuclear_projmol - lV_nuclear_projmol)
# Grid Enuc: E_ext[n] = \int n(r) v_nuc(r) dr, with v_nuc(r) = -sum_A Z_A / |r-R_A|.
v_nuc_grid = np.zeros(coords.shape[0], dtype=float)
for ia in range(mol.natm):
    rA = mol.atom_coord(ia)
    zA = mol.atom_charge(ia)
    dr = coords - rA
    dist = np.linalg.norm(dr, axis=1)
    dist = np.maximum(dist, 1e-12)
    v_nuc_grid -= zA / dist
v_nuc_grid_l = np.zeros(coords.shape[0], dtype=float)
for ia in range(lgeo1.natm):
    rA = lgeo1.atom_coord(ia)
    zA = lgeo1.atom_charge(ia)
    dr = coords - rA
    dist = np.linalg.norm(dr, axis=1)
    dist = np.maximum(dist, 1e-12)
    v_nuc_grid_l -= zA / dist
v_nuc_grid_r = np.zeros(coords.shape[0], dtype=float)
for ia in range(rgeo1.natm):
    rA = rgeo1.atom_coord(ia)
    zA = rgeo1.atom_charge(ia)
    dr = coords - rA
    dist = np.linalg.norm(dr, axis=1)
    dist = np.maximum(dist, 1e-12)
    v_nuc_grid_r -= zA / dist


Enuc_grid_l = float(np.dot(w, rho_l[0] * v_nuc_grid_l))
Enuc_grid_r = float(np.dot(w, rho_r[0] * v_nuc_grid_r))
Enuc_grid_nf = float(np.dot(w, rho_tot[0] * v_nuc_grid))
Enuc_nad_grid = Enuc_grid_nf - Enuc_grid_l - Enuc_grid_r
print("Enuc_grid_l =", Enuc_grid_l, "Enuc_grid_r =", Enuc_grid_r)
print("Enuc_grid_nf =", Enuc_grid_nf)
print("Enuc_nad_grid =", Enuc_nad_grid)


