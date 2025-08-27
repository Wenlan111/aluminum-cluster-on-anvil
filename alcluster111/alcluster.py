import time
from pyscf import gto, scf, dft, df, lib
from pyscf.dft import gen_grid
from pyscf.scf.addons import remove_linear_dep_, smearing_


#create a cluster of Al atoms
cluster = gto.Mole()
cluster.atom= '''

Al  0.00000000  0.00000000  0.00000000
Al  0.00000000 -3.29777172  2.33187675
Al  0.00000000 -6.59554425  4.66375407
Al  0.00000000 -9.89331597  6.99563081
Al -2.85595408  1.64888586  2.33187675
Al -2.85595408 -1.64888586  4.66375349
Al -2.85595408 -4.94665840  6.99563081
Al -5.71190888  3.29777213  4.66375407
Al -5.71190888  0.00000041  6.99563081
Al -8.56786296  4.94665799  6.99563081
Al  2.85595408  1.64888586  2.33187675
Al  2.85595408 -1.64888586  4.66375349
Al  2.85595408 -4.94665840  6.99563081
Al  0.00000000  3.29777172  4.66375349
Al  0.00000000  0.00000000  6.99563024
Al -2.85595479  4.94665799  6.99563081
Al  5.71190888  3.29777213  4.66375407
Al  5.71190888  0.00000041  6.99563081
Al  2.85595479  4.94665799  6.99563081
Al  8.56786296  4.94665799  6.99563081
Al -1.42797740 -0.82444313  2.33187732
Al -1.42797740 -4.12221485  4.66375407
Al -1.42797740 -7.41998739  6.99563139
Al -4.28393148  0.82444273  4.66375407
Al -4.28393148 -2.47332899  6.99563081
Al -7.13988627  2.47332899  6.99563139
Al  1.42797669  0.82444273  4.66375407
Al  1.42797669 -2.47332899  6.99563081
Al -1.42797740  2.47332859  6.99563081
Al  4.28393148  2.47332899  6.99563139
Al  1.42797740 -0.82444313  2.33187732
Al  1.42797740 -4.12221485  4.66375407
Al  1.42797740 -7.41998739  6.99563139
Al -1.42797669  0.82444273  4.66375407
Al -1.42797669 -2.47332899  6.99563081
Al -4.28393148  2.47332899  6.99563139
Al  4.28393148  0.82444273  4.66375407
Al  4.28393148 -2.47332899  6.99563081
Al  1.42797740  2.47332859  6.99563081
Al  7.13988627  2.47332899  6.99563139
Al  0.00000000  1.64888627  2.33187732
Al  0.00000000 -1.64888545  4.66375407
Al  0.00000000 -4.94665799  6.99563139
Al -2.85595408  3.29777213  4.66375407
Al -2.85595408  0.00000041  6.99563081
Al -5.71190888  4.94665840  6.99563139
Al  2.85595408  3.29777213  4.66375407
Al  2.85595408  0.00000041  6.99563081
Al -0.00000000  4.94665799  6.99563081
Al  5.71190888  4.94665840  6.99563139


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
mf.conv_tol = 5e-05  
# Metallic stability
mf = smearing_(mf, sigma=0.01, method='fermi')
mf.verbose = 4
# Coarse grid
mf.grids.level = 3
mf.grids.prune = gen_grid.treutler_prune
mf.small_rho_cutoff = 1e-6
mf.diis_space  = 12
mf.level_shift = 0.2
mf.damp        = 0.15

# Log sizes
mf.grids.build(with_non0tab=True)
e=mf.kernel()
ngrid = sum(len(b) for b in mf.grids.coords)
nao = cluster.nao_nr()
with open('./grid.dat', 'w') as f:
    f.write('total energy:' + str(e))
    f.write(f"grid points: {int(ngrid)}\n")
    f.write(f"number of orbitals: {int(nao)}\n")
