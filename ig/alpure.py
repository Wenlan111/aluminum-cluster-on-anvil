import os
import time
from pyscf import gto, scf, dft, df, lib
from pyscf.dft import gen_grid
from pyscf.scf.addons import remove_linear_dep_, smearing_
import numpy as np
import matplotlib.pyplot as plt


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

    '''

cluster.basis='6-31g'
#spherical gaussians
cluster.cart = False
#spin and charge
#cluster.spin = 2
cluster.build()
cluster.verbose = 4
# ---------- RKS object ----------
mf = scf.RHF(cluster) 
#mf.init_guess = ''
mf.chkfile = './hfpure.chk'
mf = mf.density_fit(auxbasis='weigend')   # enable RI-J first
mf = mf.apply(remove_linear_dep_)
mf.direct_scf = True
# Metallic stability 
mf = smearing_(mf, sigma=0.01, method='fermi')
mf.diis_space  = 12 
mf.level_shift = 0.5
mf.damp        = 0.15
#mf.newton()
mf.verbose = 4
mf.max_cycle   = 200
mf.conv_tol = 5e-5
e=mf.kernel()
