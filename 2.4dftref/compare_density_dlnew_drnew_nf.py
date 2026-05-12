#!/usr/bin/env python3
"""
Compare the OLD PDFT fragment density n_f (= n_l + n_r) against the NEW
DFT reference density n_ref built from the last-normal-cycle full-system
spin DMs saved by 2.4dftref/dftref.py.

Inputs (all defaults relative to this directory, i.e. 2.4dftref/):

  --checkpoint  ../pdft_checkpointnewb6.pkl
        Latest 2.4 PDFT checkpoint.  Needs Dal/Dbl/Dar/Dbr; if nl/nr
        are stored with matching length we reuse them, otherwise we
        recompute n_f on this grid from the fragment DMs.
  --checkref    al2.4_sigma0.002_last_dm.pkl
        New reference; must contain Da_last_normal / Db_last_normal
        (full-system spin DMs in the SAME basis as --xyz).
  --xyz         3.xyz
        Full Al50+O2 geometry (must be the geometry the new reference
        was solved at; for d=2.4 this is 3.xyz, charge=0, spin=2).
  --grid-level  3
        Level of the dft.gen_grid.Grids over the full molecule.  Must
        match the PDFT grid (level=3) so we can reuse nl/nr.
  --frag-dir    ..
        Where to find o.xyz and rgeo.xyz if we need to recompute n_f
        from Dal/Dbl/Dar/Dbr (only used as a fallback).
  --old-chk     "" (skip)
        Optional path to the OLD reference chk file (e.g. ../al2.4.chk).
        If given, also compute n_ref_old from it and print the same
        comparison block against the old reference.

Run e.g.:

    cd 2.4dftref
    python compare_density_dlnew_drnew_nf.py
    python compare_density_dlnew_drnew_nf.py --old-chk ../al2.4.chk
"""

from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np
from pyscf import dft, gto
from pyscf.scf import chkfile as scfchk


# ---------------------------------------------------------------------------
# Custom basis (must match 2.4capdft.py and dftref.py exactly).
# ---------------------------------------------------------------------------
_OBASIS_STR = """O    S
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

_ALBASIS_STR = """Al    S
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


def _basis_defs():
    return gto.basis.parse(_OBASIS_STR), gto.basis.parse(_ALBASIS_STR)


def _density_from_dm(mol: gto.Mole, coords: np.ndarray, dm: np.ndarray,
                     *, chunk: int = 20000) -> np.ndarray:
    """Evaluate rho(r) for a total (or summed-spin) DM, chunked over grid points."""
    dm = np.asarray(dm)
    if dm.ndim == 3:
        dm = dm[0] + dm[1]
    rho = np.empty(len(coords), dtype=np.float64)
    for s in range(0, len(coords), chunk):
        e = min(s + chunk, len(coords))
        ao = dft.numint.eval_ao(mol, coords[s:e], deriv=0)
        rho[s:e] = dft.numint.eval_rho(mol, ao, dm, xctype="LDA")
        del ao
    return rho


def _report(label_a: str, label_b: str, n_a: np.ndarray, n_b: np.ndarray,
            w: np.ndarray) -> None:
    diff = n_a - n_b
    l1 = float(np.dot(w, np.abs(diff)))
    l2 = float(np.sqrt(np.dot(w, diff * diff)))
    mx = float(np.max(np.abs(diff)))
    intd = float(np.dot(w, diff))
    print(f"  {label_a:>11s} - {label_b:<11s}  "
          f"L1={l1: .6e}  L2={l2: .6e}  max|dn|={mx: .3e}  "
          f"int(dn)={intd: .3e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", default="../pdft_checkpointnewb6.pkl",
                        help="old PDFT checkpoint with Dal/Dbl/Dar/Dbr (and ideally nl/nr)")
    parser.add_argument("--checkref", default="al2.4_sigma0.002_last_dm.pkl",
                        help="new reference pkl with Da_last_normal/Db_last_normal")
    parser.add_argument("--xyz", default="3.xyz",
                        help="full-system geometry (must match the reference)")
    parser.add_argument("--grid-level", type=int, default=3)
    parser.add_argument("--frag-dir", default="..",
                        help="dir containing o.xyz and rgeo.xyz (fallback path)")
    parser.add_argument("--old-chk", default="",
                        help="optional old reference chk for an extra comparison")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spin", type=int, default=2)
    args = parser.parse_args()

    print("=" * 78)
    for k, v in vars(args).items():
        print(f"  {k:>11s} = {v}")
    print("=" * 78)

    obasis, albasis = _basis_defs()

    # --- full-system mol & grid ------------------------------------------
    t = time.time()
    mol = gto.Mole()
    mol.atom = args.xyz
    mol.unit = "angstrom"
    mol.basis = {"O": obasis, "Al": albasis}
    mol.charge = args.charge
    mol.spin = args.spin
    mol.build()
    print(f"[mol] NAO={mol.nao_nr()}  nelec={mol.nelec}  "
          f"natoms={mol.natm}  ({time.time() - t:.1f}s)")

    t = time.time()
    grid = dft.gen_grid.Grids(mol)
    grid.level = args.grid_level
    grid.build()
    coords = grid.coords
    w = grid.weights
    ngrid = len(w)
    print(f"[grid] ngrid={ngrid}  ({time.time() - t:.1f}s)")

    # --- new reference density (from dftref pkl) -------------------------
    with open(args.checkref, "rb") as f:
        ref = pickle.load(f)
    Da_ref = np.asarray(ref["Da_last_normal"])
    Db_ref = np.asarray(ref["Db_last_normal"])
    print(f"[new ref] sigma={ref.get('sigma')}  converged={ref.get('converged')}  "
          f"last_cycle={ref.get('last_cycle')}  E_tot={ref.get('e_tot'):.10f}")
    if Da_ref.shape[-1] != mol.nao_nr():
        raise RuntimeError(
            f"new-ref DM nao={Da_ref.shape[-1]} != full mol nao={mol.nao_nr()}; "
            f"basis or xyz mismatch")
    t = time.time()
    n_ref_new = _density_from_dm(mol, coords, Da_ref + Db_ref)
    ne_ref_new = float(np.dot(w, n_ref_new))
    print(f"[new ref] N(n_ref_new) = {ne_ref_new:.6f}  ({time.time() - t:.1f}s)")
    del Da_ref, Db_ref

    # --- old PDFT n_f ----------------------------------------------------
    with open(args.checkpoint, "rb") as f:
        ckpt = pickle.load(f)
    print(f"[pdft ckpt] step={ckpt.get('step')}  Ef={ckpt.get('Ef'):.6f}  "
          f"L1(stored)={ckpt.get('L1'):.6f}")
    nl_s = ckpt.get("nl")
    nr_s = ckpt.get("nr")
    if (nl_s is not None and nr_s is not None
            and len(nl_s) == ngrid and len(nr_s) == ngrid):
        n_f = np.asarray(nl_s) + np.asarray(nr_s)
        nf_src = f"checkpoint nl+nr (len={len(nl_s)})"
    else:
        print(f"[pdft ckpt] stored nl/nr len mismatch "
              f"(have {None if nl_s is None else len(nl_s)}, need {ngrid}); "
              f"recomputing on this grid from fragment DMs")
        lgeo1 = gto.Mole()
        lgeo1.atom = os.path.join(args.frag_dir, "o.xyz")
        lgeo1.unit = "angstrom"
        lgeo1.basis = {"O": obasis, "ghost-Al": "6-31g*"}
        lgeo1.spin = 2
        lgeo1.build()
        rgeo1 = gto.Mole()
        rgeo1.atom = os.path.join(args.frag_dir, "rgeo.xyz")
        rgeo1.unit = "angstrom"
        rgeo1.basis = {"ghost-O": "6-31g*", "Al": albasis}
        rgeo1.spin = 0
        rgeo1.build()
        nl = _density_from_dm(lgeo1, coords, ckpt["Dal"] + ckpt["Dbl"])
        nr = _density_from_dm(rgeo1, coords, ckpt["Dar"] + ckpt["Dbr"])
        n_f = nl + nr
        nf_src = "recomputed from Dal/Dbl/Dar/Dbr"
    ne_f = float(np.dot(w, n_f))
    print(f"[pdft ckpt] nf source: {nf_src}")
    print(f"[pdft ckpt] N(n_f)       = {ne_f:.6f}")

    # --- (optional) old reference density --------------------------------
    n_ref_old = None
    if args.old_chk:
        t = time.time()
        _, scf_dict = scfchk.load_scf(args.old_chk)
        mf_tmp = dft.UKS(mol)
        Da_o, Db_o = mf_tmp.make_rdm1(scf_dict["mo_coeff"], scf_dict["mo_occ"])
        if Da_o.shape[-1] != mol.nao_nr():
            print(f"[old ref] WARNING: nao={Da_o.shape[-1]} != mol.nao={mol.nao_nr()}; "
                  f"skipping old-ref comparison")
        else:
            n_ref_old = _density_from_dm(mol, coords, Da_o + Db_o)
            ne_ref_old = float(np.dot(w, n_ref_old))
            print(f"[old ref] N(n_ref_old) = {ne_ref_old:.6f}  "
                  f"({time.time() - t:.1f}s)")

    # --- comparisons -----------------------------------------------------
    print()
    print("---- pairwise (a) - (b)  L1 / L2 / max|dn| / int(dn) ----")
    _report("n_f", "n_ref_new", n_f, n_ref_new, w)
    if n_ref_old is not None:
        _report("n_f", "n_ref_old", n_f, n_ref_old, w)
        _report("n_ref_new", "n_ref_old", n_ref_new, n_ref_old, w)

    print()
    print("---- interpretation ----")
    print("  L1(n_f, n_ref_new) : L1 your *new* PDFT would start at if")
    print("                       warm-started from this checkpoint.")
    if n_ref_old is not None:
        print("  L1(n_f, n_ref_old) : sanity check vs. old PDFT printed L1.")
        print("  L1(n_ref_new, n_ref_old): how much the reference target moved.")


if __name__ == "__main__":
    main()
