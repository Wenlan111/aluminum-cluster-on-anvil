import pyscf
from pyscf import gto, dft
from scipy import optimize
import fragments
import pickle
from pyscf.scf import chkfile as scfchk
import numpy as np
import scipy.special as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
#define the molecule

data = pickle.load(open("pdft2.7_checkpointnewb.pkl", "rb"))
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
mol, scf_dict = scfchk.load_scf('al2.7.chk')
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

#grid
grid = dft.gen_grid.Grids(mol)
grid.level = 3
grid.build()
coords = grid.coords
w = grid.weights
ngrid = int(coords.shape[0])
print(f"grid level=3, npoints={ngrid}", flush=True)

# ---------------------------------------------------------------------------
ao_values = dft.numint.eval_ao(mol, coords, deriv=1)
phi, phi_x, phi_y, phi_z = ao_values

print("eval_ao mol done", flush=True)
ao_values_r1 = dft.numint.eval_ao(rgeo1, coords, deriv=1)
phi_r1, phi_r1_x, phi_r1_y, phi_r1_z = ao_values_r1

print("eval_ao rgeo1 done", flush=True)
ao_values_l1 = dft.numint.eval_ao(lgeo1, coords, deriv=1)

phi_l1, phi_l1_x, phi_l1_y, phi_l1_z = ao_values_l1  

rho_r = dft.numint.eval_rho(rgeo1, ao_values_r1, Dar + Dbr, xctype="GGA")[0]
rho_l = dft.numint.eval_rho(lgeo1, ao_values_l1, Dal + Dbl, xctype="GGA")[0]

def overlap_integral(n1, n2, p=1/2):
    return np.sum((n1*n2)**p*w)
S = overlap_integral(rho_r, rho_l, 1)
print("S =", S)
def overlap_approximation(n1, n2, D, p=0.5):
    return sp.erf(overlap_integral(n1, n2, p) * 2 / D)


def diff_fn(n1, n2, d, Exc1, Exc2, D, p=0.5):
    Eoa1 = overlap_approximation(n1, n2, D, p) * Exc1
    Eoa2 = overlap_approximation(n1, n2, D, p) * Exc2
    return Eoa2 - Eoa1 - d


def fit_D(n1, n2, d, Exc1, Exc2, D0, p=0.5):
    return optimize.minimize(
        lambda x: diff_fn(n1, n2, d, Exc1, Exc2, float(x[0]), p=p),
        x0=np.array([D0], dtype=np.float64),
        method="Nelder-Mead",
    )

D_scan = np.logspace(-3, 2, 201)
p_scan = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

S_int_arr = np.zeros(len(p_scan))
S_vs_D = np.zeros((len(p_scan), len(D_scan)))

print("Scanning over log-spaced D in [1e-3, 1e2] and p in [0.5, 1.0]", flush=True)
for ip, p in enumerate(p_scan):
    S_int_arr[ip] = overlap_integral(rho_r, rho_l, p)
    print(f"p = {p:.3f}, S_int(p) = {S_int_arr[ip]:.6e}", flush=True)
    S_vs_D[ip] = sp.erf(S_int_arr[ip] * 2.0 / D_scan)

np.savez("oa_scan.npz", D=D_scan, p=p_scan, S=S_vs_D, S_int=S_int_arr)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ip, p in enumerate(p_scan):
    axes[0].plot(D_scan, S_vs_D[ip], label=f"p = {p:.2f}")
    axes[1].plot(D_scan, S_vs_D[ip], label=f"p = {p:.2f}")
axes[0].set_xscale("log")
axes[0].set_xlabel("D (log scale)")
axes[0].set_ylabel("S(D) = erf(2 * S_int(p) / D)")
axes[0].set_title("Overlap approximation S(D), log-D")
axes[0].grid(True, which="both", alpha=0.3)
axes[0].legend()

axes[1].set_xlim(1.0, 10.0)
axes[1].set_xlabel("D")
axes[1].set_ylabel("S(D)")
axes[1].set_title("S(D) on requested window D in [1, 10]")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

fig.tight_layout()
fig.savefig("oa_scan_S_vs_D.png", dpi=200)
print("Saved plot to oa_scan_S_vs_D.png", flush=True)
