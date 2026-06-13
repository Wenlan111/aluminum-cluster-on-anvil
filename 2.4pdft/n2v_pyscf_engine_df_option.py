"""
Optional Coulomb-metric density fitting for n2v PySCFEngine.get_S3.

On first import, wraps whatever ``PySCFEngine.get_S3`` is at that moment:
- In ``inversion.py``, import this module *after* assigning chunked grid ``get_S3`` so the
  fallback when ``s3_use_density_fit`` is False stays chunked.
- In ``test_s3_density_fit_h2.py``, import this module first so the fallback is stock n2v
  grid quadrature (fine for H2).

Control at runtime: ``eng.s3_use_density_fit`` and ``eng.s3_df_auxbasis``.
"""

from __future__ import annotations

import numpy as np
from pyscf import gto
from pyscf.df import addons, incore

_orig_get_S3 = None
_PATCHED = False


def _get_S3_density_fit(self):
    """S3_{μνp} via Coulomb-metric DF (aux basis), then overlap to PBS AOs."""
    mol = self.mol
    auxbasis = getattr(self, "s3_df_auxbasis", "weigend")
    rcond = float(getattr(self, "s3_df_metric_rcond", 1e-9))

    auxmol = addons.make_auxmol(mol, auxbasis)

    j3c = incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
    v = np.moveaxis(j3c, -1, 0).copy()
    del j3c

    j2c = incore.fill_2c2e(mol, auxmol, intor="int2c2e")
    inv_j = np.linalg.pinv(j2c, rcond=rcond)

    if self.pbs_str == "same":
        pbs_mol = mol
    else:
        pbs_mol = self.pbs

    sp = gto.intor_cross("int1e_ovlp", auxmol, pbs_mol)
    x = np.einsum("QP,Pp->Qp", inv_j, sp, optimize=True)
    s3 = np.einsum("Qmn,Qp->mnp", v, x, optimize=True)
    return s3


def _patched_get_S3(self):
    if getattr(self, "s3_use_density_fit", False):
        return _get_S3_density_fit(self)
    return _orig_get_S3(self)


def _apply_patch():
    global _orig_get_S3, _PATCHED
    if _PATCHED:
        return
    import n2v.engines.pyscf as _eng

    _orig_get_S3 = _eng.PySCFEngine.get_S3
    _eng.PySCFEngine.get_S3 = _patched_get_S3
    _PATCHED = True


_apply_patch()


def _set_system(eng, molecule, basis, ref=1, pbs="same", pbs_mol=None):
    """Like n2v PySCFEngine.set_system but copies charge/spin before pbs.build()."""
    eng.mol = molecule
    eng.basis_str = basis
    eng.pbs_str = pbs
    eng.ref = ref

    if pbs != "same":
        if pbs_mol is not None:
            eng.pbs = pbs_mol
        else:
            eng.pbs = gto.Mole()
            eng.pbs.atom = eng.mol.atom
            eng.pbs.basis = eng.pbs_str
            eng.pbs.charge = eng.mol.charge
            eng.pbs.spin = eng.mol.spin
            eng.pbs.build()
    else:
        eng.pbs = None

    eng.nalpha = eng.mol.nelec[0]
    eng.nbeta = eng.mol.nelec[1]


def compute_s3(
    mol,
    basis,
    *,
    pbs="same",
    pbs_mol=None,
    use_density_fit=False,
    auxbasis="weigend",
    ref=1,
):
    """Fresh PySCFEngine → get_S3() (requires this module imported so patch is active)."""
    from n2v.engines.pyscf import PySCFEngine

    eng = PySCFEngine()
    eng.s3_use_density_fit = bool(use_density_fit)
    if use_density_fit:
        eng.s3_df_auxbasis = auxbasis
    _set_system(eng, mol, basis, ref=ref, pbs=pbs, pbs_mol=pbs_mol)
    if use_density_fit:
        # DF get_S3 uses integrals only; skip PySCFGrider init (it runs a
        # spurious UKS/SVWN SCF on every mol and can trigger overlap warnings).
        eng.nbf = eng.mol.nao_nr()
        eng.npbs = eng.nbf if pbs == "same" else eng.pbs.nao_nr()
    else:
        eng.initialize()
    return eng.get_S3(), eng
