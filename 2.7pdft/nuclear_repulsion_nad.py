"""
Nuclear repulsion for 2.7 PDFT fragment geometries (same as 2.7capdft.py).

  El   = mol.energy_nuc() for lgeo1  (real O2; ghost-Al → Z=0, no contribution)
  Er   = mol.energy_nuc() for rgeo1  (real Al cluster; ghost-O → Z=0)
  Etot = mol.energy_nuc() for full geo (4.xyz: O2 + Al, no ghosts)
  Enad = Etot - El - Er   (cross O⋯Al Coulomb pairs not in isolated fragments)

Energies in Hartree unless -eV is passed.
"""

import argparse
import os
import sys

from pyscf import gto

# Basis sets and fragment mols match 2.7pdft/2.7capdft.py (do not import that file: it runs PDFT).

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


def build_molecules(_dir):
    geo = gto.Mole()
    geo.atom = os.path.join(_dir, "4.xyz")
    geo.unit = "angstrom"
    geo.basis = {"O": obasis, "Al": albasis}
    geo.charge = 0
    geo.spin = 2
    geo.build()

    lgeo1 = gto.Mole()
    lgeo1.atom = os.path.join(_dir, "o.xyz")
    lgeo1.unit = "angstrom"
    lgeo1.basis = {"O": obasis, "ghost-Al": "6-31g*"}
    lgeo1.spin = 2
    lgeo1.build()

    rgeo1 = gto.Mole()
    rgeo1.atom = os.path.join(_dir, "rgeo.xyz")
    rgeo1.unit = "angstrom"
    rgeo1.basis = {"ghost-O": "6-31g*", "Al": albasis}
    rgeo1.spin = 0
    rgeo1.build()

    return lgeo1, rgeo1, geo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-d",
        "--dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory with o.xyz, rgeo.xyz, 4.xyz (default: script dir)",
    )
    ap.add_argument("--ev", action="store_true", help="Print in eV (27.211386 Ha)")
    args = ap.parse_args()

    _dir = os.path.abspath(args.dir)
    for name in ("o.xyz", "rgeo.xyz", "4.xyz"):
        if not os.path.isfile(os.path.join(_dir, name)):
            print(f"Missing {name} in {_dir}", file=sys.stderr)
            sys.exit(1)

    lgeo1, rgeo1, geo = build_molecules(_dir)
    el = float(lgeo1.energy_nuc())
    er = float(rgeo1.energy_nuc())
    etot = float(geo.energy_nuc())
    enad = etot - el - er

    scale = 27.211386245988 if args.ev else 1.0
    unit = "eV" if args.ev else "Ha"

    def s(x):
        return f"{x * scale:.4f}"

    print(f"lgeo1 (O2 + ghost-Al)  E_nn = {s(el)} {unit}")
    print(f"rgeo1 (Al + ghost-O)   E_nn = {s(er)} {unit}")
    print(f"geo  (4.xyz full)      E_nn = {s(etot)} {unit}")
    print(f"Enad = Etot - El - Er  = {s(enad)} {unit}")


if __name__ == "__main__":
    main()
