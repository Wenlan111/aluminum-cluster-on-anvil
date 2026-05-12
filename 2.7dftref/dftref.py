"""
Single-point UKS reference calculation for the full Al50+O2 system at d=2.7 A.

The DFT solver settings match the Al ensemble (right fragment) setup used in
the pdft workflow (see ``FragmentDFT(..., metal=True, smearing=True,
newton=False)`` in ``fragments.py``):

    - dft.UKS, xc='pbe'
    - density_fit(auxbasis='weigend')
    - remove_linear_dep_(lindep=1e-4)
    - direct_scf = True, max_cycle = 200, conv_tol = 5e-5
    - grids.level = 3, grids.prune = treutler_prune, small_rho_cutoff = 1e-6
    - diis_space = 12
    - smearing_(sigma=sigma, method='fermi', fix_spin=True)
    - level_shift = 0.2, damp = 0.15

The full-system molecule mirrors the geo built in ``2.7capdft.py``:
``atom='4.xyz'``, ``basis={'O': obasis, 'Al': albasis}``, ``charge=0``,
``spin=2``.

For each sigma in [0.002, 0.005, 0.01, 0.02] we:
    1. Run UKS SCF from scratch.
    2. Capture the density matrix used to build the final Fock (the dm at the
       last normal SCF cycle), via a callback hooked into mf.callback. This
       avoids using the post-loop "extra cycle" make_rdm1(mo_coeff, mo_occ).
    3. Dump:
        - ``al2.7_sigma{sigma}.chk`` via pyscf.scf.chkfile.dump_scf (standard
          chk file, loadable with scfchk.load_scf).
        - ``al2.7_sigma{sigma}_last_dm.pkl`` with the last-normal-cycle
          (Da, Db), the matching mo_coeff/mo_occ/mo_energy at that cycle, the
          final mo_coeff/mo_occ/mo_energy, e_tot, and convergence flag.
"""

import os
import sys
import time
import pickle

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


mol = gto.Mole()
mol.atom = "4.xyz"
mol.unit = "angstrom"
mol.basis = {'O': obasis, 'Al': albasis}
mol.charge = 0
mol.spin = 2
mol.cart = False
mol.max_memory = MAX_MEMORY
mol.verbose = 4
mol.build()

print(f"NAO = {mol.nao_nr()}, nelec = {mol.nelec}", flush=True)


def make_last_cycle_callback(store):
    """Return a callback that caches dm/mo_* from the last *normal* SCF cycle.

    Mirrors fragments.FragmentDFT._cache_last_normal_dm: anything with
    envs['cycle'] < 0 (the post-loop "extra" iteration PySCF runs after
    convergence) is ignored.
    """
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


def run_one_sigma(sigma):
    print(f"\n========== sigma = {sigma:g} ==========", flush=True)
    mf = dft.UKS(mol)
    mf.xc = 'pbe'
    mf = mf.density_fit(auxbasis='weigend')
    mf = remove_linear_dep_(mf, lindep=1e-4)
    mf.direct_scf = True
    mf.max_cycle = 200
    mf.conv_tol = 5e-5
    mf.verbose = 4
    mf.grids.level = 3
    mf.grids.prune = treutler_prune
    mf.small_rho_cutoff = 1e-6
    mf.diis_space = 12
    mf.max_memory = MAX_MEMORY

    mf = smearing_(mf, sigma=sigma, method="fermi", fix_spin=(mol.spin != 0))
    mf.level_shift = 0.2
    mf.damp = 0.15

    last = {}
    mf.callback = make_last_cycle_callback(last)

    t0 = time.time()
    e_tot = mf.kernel()
    dt = time.time() - t0
    converged = bool(mf.converged)
    last_cycle = last.get("cycle", -1)

    print(
        f"[sigma={sigma:g}] E_tot = {e_tot:.10f} Eh | converged = {converged}"
        f" | last_cycle = {last_cycle} | time = {dt:.1f} s",
        flush=True,
    )

    chkname = f"al2.7_sigma{sigma:g}.chk"
    scfchk.dump_scf(
        mol, chkname,
        e_tot=e_tot,
        mo_coeff=mf.mo_coeff,
        mo_occ=mf.mo_occ,
        mo_energy=mf.mo_energy,
    )
    print(f"  -> wrote {chkname}", flush=True)

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
    }
    pklname = f"al2.7_sigma{sigma:g}_last_dm.pkl"
    with open(pklname, "wb") as f:
        pickle.dump(extra, f)
    print(f"  -> wrote {pklname}", flush=True)

    return {
        "sigma": sigma,
        "e_tot": e_tot,
        "converged": converged,
        "last_cycle": last_cycle,
        "time_s": dt,
    }


sigmas = [0.002, 0.005, 0.01, 0.02]

results = []
for sigma in sigmas:
    try:
        r = run_one_sigma(sigma)
    except Exception as exc:
        print(f"[sigma={sigma:g}] FAILED: {type(exc).__name__}: {exc}", flush=True)
        results.append({
            "sigma": sigma,
            "e_tot": float("nan"),
            "converged": False,
            "last_cycle": -1,
            "time_s": float("nan"),
            "error": f"{type(exc).__name__}: {exc}",
        })
        continue
    results.append(r)

print("\n========== Summary ==========", flush=True)
print(f"{'sigma':>8s}  {'E_tot (Eh)':>18s}  {'converged':>10s}  {'last_cyc':>9s}  {'time (s)':>10s}")
for r in results:
    print(
        f"{r['sigma']:>8.4f}  {r['e_tot']:>18.10f}  {str(r['converged']):>10s}"
        f"  {r['last_cycle']:>9d}  {r['time_s']:>10.1f}"
    )

with open("dftref_summary.pkl", "wb") as f:
    pickle.dump(results, f)
print("Wrote dftref_summary.pkl", flush=True)
