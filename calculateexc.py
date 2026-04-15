import pyscf
from pyscf import gto
from pyscf import dft
import numpy as np
import pickle
import fragments
from pyscf.scf import chkfile as scfchk


#read the pkl file
with open("pdft_checkpointnewb6.pkl", "rb") as f:
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
    """Electronic Hartree energy: 1/2 Tr(Da Ja) + 1/2 Tr(Db Jb), Coulomb via DF."""
    da = _symm(da)
    db = _symm(db)
    mfj = dft.UKS(m).density_fit(auxbasis=auxbasis)
    ja, jb = mfj.get_j(dm=[da, db])
    return 0.5 * (
        float(np.einsum("ij,ji", da, ja)) + float(np.einsum("ij,ji", db, jb))
    )


mol, scf_dict = scfchk.load_scf('al2.4.chk')
mf = dft.UKS(mol)
mo_coeff  = scf_dict['mo_coeff']
mo_occ    = scf_dict['mo_occ']
mo_energy = scf_dict['mo_energy']
Daref, Dbref = mf.make_rdm1(mo_coeff, mo_occ) 


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

rho_r = dft.numint.eval_rho(rgeo1, ao_values_r1, Dar + Dbr, xctype="GGA")
rho_l = dft.numint.eval_rho(lgeo1, ao_values_l1, Dal + Dbl, xctype="GGA")
rho_tot = rho_r + rho_l

xc = "PBE"
exc = dft.libxc.eval_xc(xc, rho_tot, spin=0)[0]
Exc = float(np.dot(w, exc * rho_tot[0]))
print("Exc =", Exc)
excl = dft.libxc.eval_xc(xc, rho_l, spin=0)[0]
excr = dft.libxc.eval_xc(xc, rho_r, spin=0)[0]
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
Ehartreetot = _hartree_uks_df(mol, Dalnew+Darnew, Dblnew+Dbrnew)
print("Ehartreel =", Ehartreel)
print("Ehartreer =", Ehartreer)
print("Ehartreetot =", Ehartreetot)
print("Ehartree nad =", Ehartreetot - Ehartreel - Ehartreer)


#kinetic energy
Tl_ao = lgeo1.intor_symmetric("int1e_kin")
Tr_ao = rgeo1.intor_symmetric("int1e_kin")
T_kineticl = float(np.einsum("ij,ji", Dal, Tl_ao) + np.einsum("ij,ji", Dbl, Tl_ao))
T_kineticr = float(np.einsum("ij,ji", Dar, Tr_ao) + np.einsum("ij,ji", Dbr, Tr_ao))
print("T_kineticl =", T_kineticl, "T_kineticr =", T_kineticr)
print("T_kinetic =", T_kineticl+T_kineticr)

#nuclear energy
V_ao = mol.intor_symmetric("int1e_nuc") 
V_l = lgeo1.intor_symmetric("int1e_nuc")
V_r = rgeo1.intor_symmetric("int1e_nuc")
V_nuclear = float(np.einsum("ij,ji", Dalnew+Darnew, V_ao) + np.einsum("ij,ji", Dblnew+Dbrnew, V_ao))
rVnuclear = float(np.einsum("ij,ji", Dar, V_r) + np.einsum("ij,ji", Dbr, V_r))
lVnuclear = float(np.einsum("ij,ji", Dal, V_l) + np.einsum("ij,ji", Dbl, V_l))

print("rVnuclear =", rVnuclear, "lVnuclear =", lVnuclear)
print("V_nuclear =", V_nuclear)
print("Vnuclearnad =", V_nuclear - rVnuclear - lVnuclear)

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


