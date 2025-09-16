import os
import time
from pyscf import gto, scf, dft, df, lib
from pyscf.dft import gen_grid
from pyscf.scf.addons import remove_linear_dep_, smearing_
import numpy as np
import matplotlib.pyplot as plt
energies = []
d_list = np.linspace(1.8, 3.0, 5)
Z_list = -4.663754 - d_list
#create a cluster of Al atoms
for Z in Z_list:
    cluster = gto.Mole()
    cluster.atom= f'''
Al   -5.711909   -3.297772   -4.663754
Al   -0.000000   -3.297772   -4.663753
Al   -2.855954    1.648886   -4.663753
Al   -2.855954   -1.648886   -2.331877
Al    5.711909   -3.297772   -4.663754
Al    2.855954    1.648886   -4.663753
Al    2.855954   -1.648886   -2.331877
Al    0.000000    6.595544   -4.663754
Al    0.000000    3.297772   -2.331877
Al    0.000000    0.000000    0.000000
Al   -1.427977   -0.824443   -4.663754
Al    4.283931   -0.824443   -4.663754
Al    1.427977    4.122215   -4.663754
Al    1.427977    0.824443   -2.331877
Al   -4.283931   -0.824443   -4.663754
Al    1.427977   -0.824443   -4.663754
Al   -1.427977    4.122215   -4.663754
Al   -1.427977    0.824443   -2.331877
Al   -2.855954   -3.297772   -4.663754
Al    2.855954   -3.297772   -4.663754
Al   -0.000000    1.648885   -4.663754
Al   -0.000000   -1.648886   -2.331877
 O   -0.000000    0.430000   {Z:.8f}
 O   -0.000000   -1.770000   {Z:.8f}

'''
#faster basis than 6-31G?
    cluster.basis='6-31G'
#spherical gaussians
    cluster.cart = False
#spin and charge
    cluster.spin = 2
    cluster.build()

# ---------- RKS object ----------
    mf = dft.UKS(cluster, xc='PBE0') 
    mf = mf.density_fit(auxbasis='weigend')   # enable RI-J first
    mf = mf.apply(remove_linear_dep_)
    mf.direct_scf = True
# Metallic stability 
    mf = smearing_(mf, sigma=0.01, method='fermi')
    mf.diis_space  = 12 
    mf.level_shift = 0.2
    mf.damp        = 0.15
    mf.verbose = 4

    mf.grids.level = 5
    mf.grids.prune = gen_grid.treutler_prune
    mf.small_rho_cutoff = 1e-6
    mf.max_cycle   = 100
    mf.conv_tol = 5e-5   
    e=mf.kernel()
    energies.append(e)


Energies = np.array(energies)
print("begin cluster2")
cluster2 = gto.Mole()
cluster2.atom= '''
Al   -5.711909   -3.297772   -4.663754
Al   -0.000000   -3.297772   -4.663753
Al   -2.855954    1.648886   -4.663753
Al   -2.855954   -1.648886   -2.331877
Al    5.711909   -3.297772   -4.663754
Al    2.855954    1.648886   -4.663753
Al    2.855954   -1.648886   -2.331877
Al    0.000000    6.595544   -4.663754
Al    0.000000    3.297772   -2.331877
Al    0.000000    0.000000    0.000000
Al   -1.427977   -0.824443   -4.663754
Al    4.283931   -0.824443   -4.663754
Al    1.427977    4.122215   -4.663754
Al    1.427977    0.824443   -2.331877
Al   -4.283931   -0.824443   -4.663754
Al    1.427977   -0.824443   -4.663754
Al   -1.427977    4.122215   -4.663754
Al   -1.427977    0.824443   -2.331877
Al   -2.855954   -3.297772   -4.663754
Al    2.855954   -3.297772   -4.663754
Al   -0.000000    1.648885   -4.663754
Al   -0.000000   -1.648886   -2.331877

'''
#faster basis than 6-31G?
cluster2.basis='6-31G'
#spherical gaussians
cluster2.cart = False
#spin and charge
cluster2.build()

# ---------- RKS object ----------
mf2 = dft.RKS(cluster2, xc='PBE0') 
mf2 = mf2.density_fit(auxbasis='weigend')   # enable RI-J first
mf2 = mf2.apply(remove_linear_dep_)
mf2.direct_scf = True
# Metallic stability 
mf2 = smearing_(mf2, sigma=0.01, method='fermi')
mf2.diis_space  = 12 
mf2.level_shift = 0.2
mf2.damp        = 0.15
mf2.verbose = 4
mf2.grids.level = 5
mf2.grids.prune = gen_grid.treutler_prune
mf2.small_rho_cutoff = 1e-6
mf2.max_cycle   = 100
mf2.conv_tol = 5e-5   
e2=mf2.kernel()
energies2.append(e2)


np.savetxt("AlclusterPBE0.dat",
           np.column_stack((d_list, E)),
           header="d(Å)  energies2(eV)",
           fmt="%.6f")
