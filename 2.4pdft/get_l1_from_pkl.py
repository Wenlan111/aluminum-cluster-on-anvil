#!/usr/bin/env python3
"""
Read optimizer output (default: h2pluspdft.pkl) and report L1 density error:

    L1 = sum_p w(p) * |nf(r_p) - nref(r_p)|

Same definition as 2.4capdft.py. Recomputes on the DFT grid from Dal/Dbl/Dar/Dbr
if L1 is not already stored in the pickle.
"""
import argparse
import os
import pickle
import sys

import numpy as np
from pyscf import dft, gto

# Basis blocks match h2pdft.py
OBASIS = gto.basis.parse(
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

ALBASIS = gto.basis.parse(
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


def dm_to_grid_density(dm, phi):
    """n(r_p) = sum_{uv} phi_u phi_v D_uv (same as h2pdft.py)."""
    dm = np.asarray(dm, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    return np.einsum("pu,pv,uv->p", phi, phi, dm, optimize=True)


def find_geo_dir():
    """h2pdft.py expects 3.xyz in cwd; on Anvil they live in parent newbasis/."""
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.getcwd(), here, os.path.join(here, "..")):
        if os.path.isfile(os.path.join(d, "3.xyz")):
            return os.path.abspath(d)
    return os.getcwd()


def build_mols(geo_dir=None):
    if geo_dir is None:
        geo_dir = find_geo_dir()
    geo = gto.Mole()
    geo.atom = os.path.join(geo_dir, "3.xyz")
    geo.unit = "angstrom"
    geo.basis = {"O": OBASIS, "Al": ALBASIS}
    geo.charge = 0
    geo.spin = 2
    geo.build()

    lgeo1 = gto.Mole()
    lgeo1.atom = os.path.join(geo_dir, "o.xyz")
    lgeo1.unit = "angstrom"
    lgeo1.basis = {"O": OBASIS, "ghost-Al": "6-31g*"}
    lgeo1.spin = 2
    lgeo1.build()

    rgeo1 = gto.Mole()
    rgeo1.atom = os.path.join(geo_dir, "rgeo.xyz")
    rgeo1.unit = "angstrom"
    rgeo1.basis = {"ghost-O": "6-31g*", "Al": ALBASIS}
    rgeo1.spin = 0
    rgeo1.build()

    return geo, lgeo1, rgeo1


def stored_l1(data):
    for key in ("L1", "L1_density_diff"):
        if key in data:
            return float(data[key]), key
    return None, None


def recompute_l1(data, grid_level, ref_dm_path, geo_dir=None):
    dal = np.asarray(data["Dal"], dtype=np.float64)
    dbl = np.asarray(data["Dbl"], dtype=np.float64)
    dar = np.asarray(data["Dar"], dtype=np.float64)
    dbr = np.asarray(data["Dbr"], dtype=np.float64)

    if "Dref" in data and data["Dref"] is not None:
        dref = np.asarray(data["Dref"], dtype=np.float64)
    else:
        with open(ref_dm_path, "rb") as f:
            ref_data = pickle.load(f)
        dref = np.asarray(
            ref_data["Da_last_normal"] + ref_data["Db_last_normal"], dtype=np.float64
        )

    geo, lgeo1, rgeo1 = build_mols(geo_dir)
    grid = dft.gen_grid.Grids(geo)
    grid.level = grid_level
    grid.build()
    w = grid.weights
    coords = grid.coords

    phi_ref = dft.numint.eval_ao(geo, coords, deriv=0)
    phi_l1 = dft.numint.eval_ao(lgeo1, coords, deriv=0)
    phi_r1 = dft.numint.eval_ao(rgeo1, coords, deriv=0)

    if "nl" in data and "nr" in data and len(data["nl"]) == len(w):
        nf = np.asarray(data["nl"], dtype=np.float64) + np.asarray(data["nr"], dtype=np.float64)
        nf_src = "pickle nl+nr"
    else:
        nl = dm_to_grid_density(dal + dbl, phi_l1)
        nr = dm_to_grid_density(dar + dbr, phi_r1)
        nf = nl + nr
        nf_src = "recomputed from Dal/Dbl/Dar/Dbr"

    if "nref" in data and len(data["nref"]) == len(w):
        nref = np.asarray(data["nref"], dtype=np.float64)
        nref_src = "pickle nref"
    else:
        nref = dm_to_grid_density(dref, phi_ref)
        nref_src = "recomputed from Dref"

    dn = nf - nref
    l1 = float(np.dot(w, np.abs(dn)))
    n_nf = float(np.dot(w, nf))
    n_ref = float(np.dot(w, nref))
    signed = float(np.dot(w, dn))

    return {
        "L1": l1,
        "N_nf": n_nf,
        "N_nref": n_ref,
        "Int_w_nf_minus_nref": signed,
        "nf_source": nf_src,
        "nref_source": nref_src,
        "ngrid": len(w),
    }


def main():
    parser = argparse.ArgumentParser(description="L1 = sum w|nf-nref| from h2pluspdft.pkl")
    parser.add_argument("--pkl", default="pdft_optimizer.pkl", help="optimizer output pickle")
    parser.add_argument("--ref-dm", default="../2.4dftref/al2.4_sigma0.002_last_dm.pkl")
    parser.add_argument("--grid-level", type=int, default=3)
    parser.add_argument("--geo-dir", default=None, help="directory with 3.xyz, o.xyz, rgeo.xyz")
    args = parser.parse_args()

    geo_dir = args.geo_dir or find_geo_dir()

    try:
        with open(args.pkl, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"error: {args.pkl} not found", file=sys.stderr)
        sys.exit(1)

    print(f"pickle: {args.pkl}")
    if "iter" in data:
        print(f"iter: {data['iter']}")
    if "step" in data:
        print(f"step: {data['step']}")
    if "L" in data:
        print(f"L (Lagrangian): {float(data['L']):.8f}")
    if "Ef" in data:
        print(f"Ef: {float(data['Ef']):.8f}")

    l1_stored, l1_key = stored_l1(data)
    if l1_stored is not None:
        print(f"L1 (stored as {l1_key}): {l1_stored:.10e}")

    stats = recompute_l1(data, args.grid_level, args.ref_dm, geo_dir)
    print(f"geo dir: {geo_dir}")
    print(f"grid level: {args.grid_level}, ngrid={stats['ngrid']}")
    print(f"nf: {stats['nf_source']}")
    print(f"nref: {stats['nref_source']}")
    print(f"N(nf)   = {stats['N_nf']:.10f}")
    print(f"N(nref) = {stats['N_nref']:.10f}")
    print(f"Int w(nf-nref) = {stats['Int_w_nf_minus_nref']:.10e}")
    print(f"L1 (recomputed) = {stats['L1']:.10e}")

    if l1_stored is not None:
        diff = abs(l1_stored - stats["L1"])
        print(f"|L1_stored - L1_recomputed| = {diff:.10e}")


if __name__ == "__main__":
    main()
