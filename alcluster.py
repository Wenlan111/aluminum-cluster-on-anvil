import time
from pyscf import gto, scf, dft, df, lib
from pyscf.dft import gen_grid
from pyscf.scf.addons import remove_linear_dep_, smearing_


#create a cluster of Al atoms
cluster = gto.Mole()
cluster.atom= '''

Al    0.000000    0.000000    0.000000
Al    0.000000    0.000000    4.038929
Al    0.000000    0.000000    8.077859
Al    0.000000    0.000000   12.116788
Al    0.000000    4.038929    0.000000
Al    0.000000    4.038929    4.038929
Al    0.000000    4.038929    8.077859
Al    0.000000    8.077859    0.000000
Al    0.000000    8.077859    4.038929
Al    0.000000   12.116788    0.000000
Al    4.038929    0.000000    0.000000
Al    4.038929    0.000000    4.038929
Al    4.038929    0.000000    8.077859
Al    4.038929    4.038929    0.000000
Al    4.038929    4.038929    4.038929
Al    4.038929    8.077859    0.000000
Al    8.077859    0.000000    0.000000
Al    8.077859    0.000000    4.038929
Al    8.077859    4.038929    0.000000
Al   12.116788    0.000000    0.000000
Al    0.000000    2.019465    2.019465
Al    0.000000    2.019465    6.058394
Al    0.000000    2.019465   10.097324
Al    0.000000    6.058394    2.019465
Al    0.000000    6.058394    6.058394
Al    0.000000   10.097324    2.019465
Al    4.038929    2.019465    2.019465
Al    4.038929    2.019465    6.058394
Al    4.038929    6.058394    2.019465
Al    8.077859    2.019465    2.019465
Al    2.019465    0.000000    2.019465
Al    2.019465    0.000000    6.058394
Al    2.019465    0.000000   10.097324
Al    2.019465    4.038929    2.019465
Al    2.019465    4.038929    6.058394
Al    2.019465    8.077859    2.019465
Al    6.058394    0.000000    2.019465
Al    6.058394    0.000000    6.058394
Al    6.058394    4.038929    2.019465
Al   10.097324    0.000000    2.019465
Al    2.019465    2.019465    0.000000
Al    2.019465    2.019465    4.038929
Al    2.019465    2.019465    8.077859
Al    2.019465    6.058394    0.000000
Al    2.019465    6.058394    4.038929
Al    2.019465   10.097324    0.000000
Al    6.058394    2.019465    0.000000
Al    6.058394    2.019465    4.038929
Al    6.058394    6.058394    0.000000
Al   10.097324    2.019465    0.000000

'''
#faster basis than 6-31G?
cluster.basis='6-31G'
#spherical gaussians
cluster.cart = False
#keep integrals in RAM
cluster.incore_anyway = True
cluster.build()

# ---------- RKS object ----------
mf = dft.RKS(cluster, xc='PBE')
mf = mf.density_fit(auxbasis='weigend')   # enable RI-J first
mf = mf.apply(remove_linear_dep_)
mf.direct_scf = True
mf.max_cycle = 100
mf.conv_tol = 1e-07  
# Metallic stability
mf = smearing_(mf, sigma=0.01, method='fermi')
mf.diis_space  = 12
mf.level_shift = 0.2
mf.damp        = 0.15
mf.verbose = 4
# Coarse grid
mf.grids.level = 4
mf.grids.prune = gen_grid.treutler_prune
mf.small_rho_cutoff = 1e-6

# Log sizes
mf.grids.build(with_non0tab=True)
ngrid = sum(len(b) for b in mf.grids.coords)
lib.logger.info(mf, "NAO = %d ; grid points ≈ %d", cluster.nao_nr(), ngrid)

e = mf.kernel()
lib.logger.info(mf, "Coarse SCF energy = %.12f Eh", e)

