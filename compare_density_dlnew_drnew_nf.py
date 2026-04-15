#!/usr/bin/env python3
"""
Compare density from Dlnew/Drnew against nf on the same grid.

Default inputs match the current PDFT workflow:
  - checkpoint: pdft_checkpointnewb6.pkl (needs Dal/Dbl/Dar/Dbr; optionally nl/nr)
  - reference chk: al2.4.chk
  - geometry files: o.xyz, rgeo.xyz, onew.xyz
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np
from pyscf import dft, gto
from pyscf.scf import chkfile as scfchk


def _basis_defs():
    obasis = gto.basis.parse(
        """O    S
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
      1.292                  1.000000"""
    )
    albasis = gto.basis.parse(
        """Al    S
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
      0.3250000              1.0000000"""
    )
    return obasis, albasis


def _project_one(d_old: np.ndarray, s: np.ndarray, t: np.ndarray) -> np.ndarray:
    m = t.T @ d_old @ t
    x = np.linalg.solve(s, m)
    return np.linalg.solve(s, x.T).T


def _density_from_dm(mol: gto.Mole, coords: np.ndarray, dm: np.ndarray) -> np.ndarray:
    ao = dft.numint.eval_ao(mol, coords, deriv=0)
    return dft.numint.eval_rho(mol, ao, dm, xctype="LDA")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="pdft_checkpointnewb5.pkl")
    parser.add_argument("--chk", default="al2.4.chk")
    parser.add_argument("--grid-level", type=int, default=3)
    args = parser.parse_args()

    with open(args.checkpoint, "rb") as f:
        data = pickle.load(f)

    dal = data["Dal"]
    dbl = data["Dbl"]
    dar = data["Dar"]
    dbr = data["Dbr"]

    obasis, albasis = _basis_defs()

    mol, _ = scfchk.load_scf(args.chk)

    lgeo1 = gto.Mole()
    lgeo1.atom = "o.xyz"
    lgeo1.unit = "angstrom"
    lgeo1.basis = {"O": obasis, "ghost-Al": "6-31g*"}
    lgeo1.spin = 2
    lgeo1.build()

    rgeo1 = gto.Mole()
    rgeo1.atom = "rgeo.xyz"
    rgeo1.unit = "angstrom"
    rgeo1.basis = {"ghost-O": "6-31g*", "Al": albasis}
    rgeo1.spin = 0
    rgeo1.build()

    lgeonew = gto.Mole()
    lgeonew.atom = "onew.xyz"
    lgeonew.unit = "angstrom"
    lgeonew.basis = {"O": obasis, "ghost-Al": albasis}
    lgeonew.build()

    rgeonew = gto.Mole()
    rgeonew.atom = "rgeo.xyz"
    rgeonew.unit = "angstrom"
    rgeonew.basis = {"ghost-O": obasis, "Al": albasis}
    rgeonew.build()

    sl = lgeonew.intor("int1e_ovlp")
    tl = gto.intor_cross("int1e_ovlp", lgeo1, lgeonew)
    sr = rgeonew.intor("int1e_ovlp")
    tr = gto.intor_cross("int1e_ovlp", rgeo1, rgeonew)

    dalnew = _project_one(dal, sl, tl)
    dblnew = _project_one(dbl, sl, tl)
    darnew = _project_one(dar, sr, tr)
    dbrnew = _project_one(dbr, sr, tr)

    dlnew = dalnew + dblnew
    drnew = darnew + dbrnew

    grid = dft.gen_grid.Grids(mol)
    grid.level = args.grid_level
    grid.build()
    coords = grid.coords
    w = grid.weights

    n_from_new = _density_from_dm(mol, coords, dlnew + drnew)

    # nf from checkpoint preferred; if not present, reconstruct on same grid from old fragment DMs.
    if "nl" in data and "nr" in data and len(data["nl"]) == len(w):
        nf = np.asarray(data["nl"]) + np.asarray(data["nr"])
        nf_src = "checkpoint nl+nr"
    else:
        nl = _density_from_dm(lgeo1, coords, dal + dbl)
        nr = _density_from_dm(rgeo1, coords, dar + dbr)
        nf = nl + nr
        nf_src = "recomputed from Dal/Dbl/Dar/Dbr"

    diff = n_from_new - nf
    ne_new = float(np.dot(w, n_from_new))
    ne_nf = float(np.dot(w, nf))
    l1 = float(np.dot(w, np.abs(diff)))
    l2 = float(np.sqrt(np.dot(w, diff * diff)))
    max_abs = float(np.max(np.abs(diff)))
    norm_diff = float(np.linalg.norm(diff))

    print(f"checkpoint: {args.checkpoint}")
    print(f"chk: {args.chk}")
    print(f"grid level: {args.grid_level}, ngrid={len(w)}")
    print(f"nf source: {nf_src}")
    print(f"N(Dlnew+Drnew) = {ne_new:.10f}")
    print(f"N(nf)          = {ne_nf:.10f}")
    print(f"L1             = {l1:.10e}")
    print(f"L2             = {l2:.10e}")
    print(f"max|dn|        = {max_abs:.10e}")
    print(f"norm(dn)       = {norm_diff:.10e}")


if __name__ == "__main__":
    main()
