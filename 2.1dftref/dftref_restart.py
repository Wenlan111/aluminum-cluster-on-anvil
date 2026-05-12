"""
Restart the un-converged Al50+O2 UKS reference at d=2.1 A, sigma=0.002.

The original ``dftref.py`` run for 2.1 (job 17144924) hit ``max_cycle = 200``
without converging:

    cycle 199:  delta_E = -2.6e-6 Eh   (good; below conv_tol = 1e-6 not quite met)
    cycle 199:  |g|     =  2.2e-3       (above default conv_tol_grad ~ 7e-4)
    cycle 199:  |ddm|   =  4-15         (oscillating)

Energy is at micro-Hartree level, but the density matrix is still bouncing
around.  We restart SCF with the saved last-normal-cycle DM as initial
guess, and apply heavier convergence aids:

    - max_cycle      : 200  -> 400
    - damp           : 0.15 -> 0.30      (heavier mixing)
    - level_shift    : 0.20 -> 0.30      (stronger virtual shift)
    - diis_space     : 12   -> 16
    - conv_tol_grad  : default -> 1e-3   (loosen gradient tol slightly,
                                          matching what we've actually seen
                                          for the other distances)

We write to NEW filenames so the originals are preserved:

    al2.1_sigma0.002_restart.chk
    al2.1_sigma0.002_restart_last_dm.pkl

Pass ``--from-restart`` on the CLI to chain another restart (uses the new
pkl as the warm-start input).
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
import psutil

from pyscf import gto, dft, lib
from pyscf.scf import chkfile as scfchk
from pyscf.scf.addons import remove_linear_dep_, smearing_
from pyscf.dft.gen_grid import treutler_prune


MAX_MEMORY = int(psutil.virtual_memory().available / 1e6)
print(f"pyscf max mem (MB) = {MAX_MEMORY}", flush=True)


obasis = gto.basis.parse('''O    S
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
      1.292                  1.000000''')

albasis = gto.basis.parse('''Al    S
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
      0.3250000              1.0000000''')


def build_mol():
    mol = gto.Mole()
    mol.atom = "2.xyz"
    mol.unit = "angstrom"
    mol.basis = {'O': obasis, 'Al': albasis}
    mol.charge = 0
    mol.spin = 2
    mol.cart = False
    mol.max_memory = MAX_MEMORY
    mol.verbose = 4
    mol.build()
    print(f"NAO = {mol.nao_nr()}, nelec = {mol.nelec}", flush=True)
    return mol


def make_last_cycle_callback(store):
    """Cache dm/mo_* from each last *normal* SCF cycle (envs['cycle'] >= 0)."""
    def cb(envs):
        cyc = envs.get("cycle", None)
        if cyc is None or cyc < 0:
            return
        dm = envs.get("dm", None)
        if dm is None:
            return
        arr = np.asarray(dm)
        if arr.ndim == 3 and arr.shape[0] == 2:
            Da, Db = arr[0], arr[1]
        elif arr.ndim == 2:
            Da = Db = 0.5 * arr
        else:
            return
        store["cycle"] = int(cyc)
        store["Da"] = np.array(Da, copy=True)
        store["Db"] = np.array(Db, copy=True)
        for k in ("mo_coeff", "mo_occ", "mo_energy", "fock"):
            v = envs.get(k, None)
            if v is not None:
                store[k] = np.array(v, copy=True)
    return cb


def load_warm_start_dm(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    Da = np.asarray(data["Da_last_normal"])
    Db = np.asarray(data["Db_last_normal"])
    print(f"[warm-start] from {pkl_path}", flush=True)
    print(f"             sigma={data.get('sigma')}  "
          f"converged={data.get('converged')}  "
          f"last_cycle={data.get('last_cycle')}  "
          f"E_tot={data.get('e_tot'):.10f}", flush=True)
    print(f"             dm shapes: Da={Da.shape}  Db={Db.shape}", flush=True)
    return np.stack([Da, Db], axis=0)


def run_restart(mol, dm0, sigma, *, max_cycle, damp, level_shift,
                diis_space, conv_tol, conv_tol_grad, out_chk, out_pkl):
    print(f"\n========== sigma = {sigma:g}  (restart) ==========", flush=True)
    print(f"  max_cycle      = {max_cycle}", flush=True)
    print(f"  damp           = {damp}", flush=True)
    print(f"  level_shift    = {level_shift}", flush=True)
    print(f"  diis_space     = {diis_space}", flush=True)
    print(f"  conv_tol       = {conv_tol}", flush=True)
    print(f"  conv_tol_grad  = {conv_tol_grad}", flush=True)

    mf = dft.UKS(mol)
    mf.xc = 'pbe'
    mf = mf.density_fit(auxbasis='weigend')
    mf = remove_linear_dep_(mf, lindep=1e-4)
    mf.direct_scf = True
    mf.max_cycle = max_cycle
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = conv_tol_grad
    mf.verbose = 4
    mf.grids.level = 3
    mf.grids.prune = treutler_prune
    mf.small_rho_cutoff = 1e-6
    mf.diis_space = diis_space
    mf.max_memory = MAX_MEMORY

    mf = smearing_(mf, sigma=sigma, method="fermi", fix_spin=(mol.spin != 0))
    mf.level_shift = level_shift
    mf.damp = damp

    last = {}
    mf.callback = make_last_cycle_callback(last)

    t0 = time.time()
    e_tot = mf.kernel(dm0=dm0)
    dt = time.time() - t0
    converged = bool(mf.converged)
    last_cycle = last.get("cycle", -1)

    print(
        f"[sigma={sigma:g}] E_tot = {e_tot:.10f} Eh | converged = {converged}"
        f" | last_cycle = {last_cycle} | time = {dt:.1f} s",
        flush=True,
    )

    scfchk.dump_scf(
        mol, out_chk,
        e_tot=e_tot,
        mo_coeff=mf.mo_coeff,
        mo_occ=mf.mo_occ,
        mo_energy=mf.mo_energy,
    )
    print(f"  -> wrote {out_chk}", flush=True)

    extra = {
        "sigma": sigma,
        "converged": converged,
        "e_tot": e_tot,
        "last_cycle": last_cycle,
        "Da_last_normal": last.get("Da"),
        "Db_last_normal": last.get("Db"),
        "mo_coeff_last_normal": last.get("mo_coeff"),
        "mo_occ_last_normal": last.get("mo_occ"),
        "mo_energy_last_normal": last.get("mo_energy"),
        "mo_coeff_final": mf.mo_coeff,
        "mo_occ_final": mf.mo_occ,
        "mo_energy_final": mf.mo_energy,
        "restart_settings": {
            "max_cycle": max_cycle, "damp": damp,
            "level_shift": level_shift, "diis_space": diis_space,
            "conv_tol": conv_tol, "conv_tol_grad": conv_tol_grad,
        },
    }
    with open(out_pkl, "wb") as f:
        pickle.dump(extra, f)
    print(f"  -> wrote {out_pkl}", flush=True)

    return {
        "sigma": sigma,
        "e_tot": e_tot,
        "converged": converged,
        "last_cycle": last_cycle,
        "time_s": dt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-pkl",
                    default="al2.1_sigma0.002_last_dm.pkl",
                    help="pkl with Da_last_normal / Db_last_normal to warm-start from")
    ap.add_argument("--sigma", type=float, default=0.002)
    ap.add_argument("--max-cycle", type=int, default=400)
    ap.add_argument("--damp", type=float, default=0.30)
    ap.add_argument("--level-shift", type=float, default=0.30)
    ap.add_argument("--diis-space", type=int, default=16)
    ap.add_argument("--conv-tol", type=float, default=1e-6)
    ap.add_argument("--conv-tol-grad", type=float, default=1e-3)
    ap.add_argument("--out-chk", default="al2.1_sigma0.002_restart.chk")
    ap.add_argument("--out-pkl", default="al2.1_sigma0.002_restart_last_dm.pkl")
    args = ap.parse_args()

    if not os.path.exists(args.input_pkl):
        sys.exit(f"input pkl not found: {args.input_pkl}")

    mol = build_mol()
    dm0 = load_warm_start_dm(args.input_pkl)

    r = run_restart(
        mol, dm0, args.sigma,
        max_cycle=args.max_cycle, damp=args.damp,
        level_shift=args.level_shift, diis_space=args.diis_space,
        conv_tol=args.conv_tol, conv_tol_grad=args.conv_tol_grad,
        out_chk=args.out_chk, out_pkl=args.out_pkl,
    )

    print("\n========== Summary ==========", flush=True)
    print(f"{'sigma':>8s}  {'E_tot (Eh)':>18s}  {'converged':>10s}  "
          f"{'last_cyc':>9s}  {'time (s)':>10s}")
    print(
        f"{r['sigma']:>8.4f}  {r['e_tot']:>18.10f}  {str(r['converged']):>10s}"
        f"  {r['last_cycle']:>9d}  {r['time_s']:>10.1f}"
    )

    with open("dftref_restart_summary.pkl", "wb") as f:
        pickle.dump([r], f)
    print("Wrote dftref_restart_summary.pkl", flush=True)


if __name__ == "__main__":
    main()
