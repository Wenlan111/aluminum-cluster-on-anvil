"""
Read inversion_ks_for_energy.pkl and report T, Vne from Da/Db.

AO basis must match inversion.py: inv.set_system(..., "6-311++G(d,p)", ...).
"""
import pickle

import numpy as np
from pyscf.scf import chkfile as scfchk

# Same main AO basis as inversion.py (must match or Da/Db shapes will not fit mol).
AO_BASIS = "6-311++G(d,p)"
CHK = "al2.4.chk"
PKL = "inversion_ks_for_energy.pkl"

mol, _scf_dict = scfchk.load_scf(CHK)
mol.basis = AO_BASIS
mol.build()

with open(PKL, "rb") as f:
    data = pickle.load(f)

Da = data["Da"]
Db = data["Db"]
Coca = data["Coca"]
Cocb = data["Cocb"]
nao = mol.nao_nr()
Dai = Coca @ Coca.T
Dbi = Cocb @ Cocb.T
if Da.shape != (nao, nao) or Db.shape != (nao, nao):
    raise ValueError(
        f"Density shape {Da.shape} vs mol nao={nao}. "
        f"Fix AO_BASIS / CHK to match the inversion run."
    )
#number of electrons
S = mol.intor("int1e_ovlp")
Ne_inv = np.einsum("ij,ji", Da + Db, S)
Ne_inv_i = np.einsum("ij,ji", Dai + Dbi, S)
print("Ne_inv =", Ne_inv)
print("Ne_inv_i =", Ne_inv_i)
T_ao = mol.intor_symmetric("int1e_kin")
T_kinetic = float(np.einsum("ij,ji", Da, T_ao) + np.einsum("ij,ji", Db, T_ao))
print("T (electronic kinetic, Ha) =", T_kinetic)

V_ao = mol.intor_symmetric("int1e_nuc")
V_nuclear = float(np.einsum("ij,ji", Da, V_ao) + np.einsum("ij,ji", Db, V_ao))
print("V (nuclear attraction, Ha) =", V_nuclear)

# Not the full molecular electronic energy (no Hartree / exchange / correlation).
T_plus_Vne = T_kinetic + V_nuclear
print("T + Vne (Ha) =", T_plus_Vne)
