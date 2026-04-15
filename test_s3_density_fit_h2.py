#!/usr/bin/env python3
"""
Compare n2v PySCFEngine.get_S3: stock grid quadrature vs Coulomb-metric DF (H2, small basis).

Run from the directory that contains this script (or ensure n2v / PySCF are on PYTHONPATH):

    python test_s3_density_fit_h2.py

Import n2v_pyscf_engine_df_option before building engines so get_S3 is wrapped
(stock grid vs DF); H2 is small so stock grid is fine.
"""

from __future__ import annotations

import sys

import numpy as np
from pyscf import gto

import n2v_pyscf_engine_df_option as s3m  # applies get_S3 wrapper on import


def main() -> int:
    mol = gto.Mole()
    mol.atom = "H 0 0 0; H 0 0 0.74"
    mol.unit = "Bohr"
    mol.basis = "sto-3g"
    mol.build()

    basis = "sto-3g"
    auxbasis = "weigend"

    s_grid, _ = s3m.compute_s3(mol, basis, pbs="same", use_density_fit=False)
    s_df, _ = s3m.compute_s3(
        mol, basis, pbs="same", use_density_fit=True, auxbasis=auxbasis
    )

    diff = s_grid - s_df
    fro = float(np.linalg.norm(diff.ravel()))
    maxabs = float(np.max(np.abs(diff)))

    print("H2, AO basis sto-3g, pbs='same'")
    print(f"  nao = {mol.nao_nr()}")
    print(f"  S3 shape (grid) = {s_grid.shape}")
    print(f"  S3 shape (DF)   = {s_df.shape}")
    print(f"  ||S3_grid - S3_DF||_F = {fro:.6e}")
    print(f"  max|S3_grid - S3_DF|  = {maxabs:.6e}")

    # Second case: separate potential basis (common in Wu–Yang).
    try:
        s_g2, _ = s3m.compute_s3(mol, basis, pbs="def2-SVP", use_density_fit=False)
        s_d2, _ = s3m.compute_s3(
            mol, basis, pbs="def2-SVP", use_density_fit=True, auxbasis=auxbasis
        )
        d2 = s_g2 - s_d2
        print()
        print("H2, AO sto-3g, pbs='def2-SVP'")
        print(f"  S3 shape (grid) = {s_g2.shape}")
        print(f"  ||Δ||_F = {float(np.linalg.norm(d2.ravel())):.6e}")
    except Exception as exc:
        print()
        print(f"(Optional pbs=def2-SVP test skipped: {exc})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
