import os
import time
from pyscf import gto, scf, dft, df, lib
from pyscf.dft import gen_grid
from pyscf.scf.addons import remove_linear_dep_, smearing_


#create a cluster of Al atoms
cluster = gto.Mole()
cluster.atom= '''

Al   -8.077859    0.000000    0.000000
Al   -4.038929   -4.038929    0.000000
Al   -4.038929   -0.000000   -4.038929
Al   -4.038929    0.000000    0.000000
Al   -0.000000   -8.077859    0.000000
Al   -0.000000   -4.038929   -4.038929
Al   -0.000000   -4.038929    0.000000
Al   -0.000000   -0.000000   -8.077859
Al   -0.000000   -0.000000   -4.038929
Al    0.000000    0.000000    0.000000
Al   -4.038929   -2.019465   -2.019465
Al   -0.000000   -6.058394   -2.019465
Al   -0.000000   -2.019465   -6.058394
Al   -0.000000   -2.019465   -2.019465
Al   -6.058394   -0.000000   -2.019465
Al   -2.019465   -4.038929   -2.019465
Al   -2.019465   -0.000000   -6.058394
Al   -2.019465   -0.000000   -2.019465
Al   -6.058394   -2.019465    0.000000
Al   -2.019465   -6.058394    0.000000
Al   -2.019465   -2.019465   -4.038929
Al   -2.019465   -2.019465    0.000000
'''
#faster basis than 6-31G?
cluster.basis='6-31G'
#spherical gaussians
cluster.cart = False
cluster.build()

# ---------- RKS object ----------
mf = dft.RKS(cluster, xc='PBE')
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
mf.conv_tol = 1e-6   
mf.kernel()
