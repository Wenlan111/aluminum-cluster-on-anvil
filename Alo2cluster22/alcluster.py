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
    mf = dft.UKS(cluster, xc='PBE') 
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
    mf.max_cycle   = 200
    mf.conv_tol = 5e-5  
    e=mf.kernel()
    energies.append(e)


mol = gto.Mole()
mol.atom= ''' 
O 0   0.43   0
O 0   1.77   0
'''
mol.basis ='cc-pvdz'
mol.spin = 2
mol.build()
mfo2 = dft.UKS(mol,'xc=PBE')
eo2 = mfo2.kernel()

Energies = np.array(energies)
E_ads = (Energies - eo2 -5330.31811937795)/27.2


np.savetxt("adsorption_energyPBE.dat",
           np.column_stack((d_list, E_ads)),
           header="d(Å)  E_ads(eV)",
           fmt="%.6f")
