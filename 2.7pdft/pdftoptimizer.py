"""
PDFT partition optimizer for Al50+O2 at 2.4 A (PBS + L-BFGS-B).

Same fragment setup as ``2.4capdft.py``. The optimization variable is a PBS
coefficient vector ``b`` (length ``npbs``). ``2.4capdft.py`` stores ``vp[p]`` on
the DFT grid; this script projects that grid vector onto PBS to obtain ``b0``.

Workflow
--------
1. Build fragments, grid, and S3 tensors (PBS defined on full ``geo``).
2. Warm-start: load capdft pickle → initial ``b0`` and ``dm_ig1``/``dm_ig2`` on ``rdft1``/``rdft2``.
3. Phase 1: damped gradient ascent on ``b`` (push into a safe region for SCF).
4. Phase 2: L-BFGS-B on ``b``; phase-1 step 1 SCF uses ``dm_ig1``/``dm_ig2`` as ``dm0``.
"""
import os
import pickle
import time

import numpy as np
import psutil
from pyscf import dft, gto
from scipy.optimize import minimize

import fragments
import n2v_pyscf_engine_df_option as s3mod

MAX_MEMORY = int(psutil.virtual_memory().available / 1e6)
print("pyscf max mem", MAX_MEMORY)

from pyscf import lib

nthreads = int(os.environ.get("SLURM_CPUS_PER_TASK", "64"))
lib.num_threads(nthreads)

print("SLURM_CPUS_PER_TASK =", os.environ.get("SLURM_CPUS_PER_TASK"), flush=True)
print("OMP_NUM_THREADS =", os.environ.get("OMP_NUM_THREADS"), flush=True)
print("MKL_NUM_THREADS =", os.environ.get("MKL_NUM_THREADS"), flush=True)
print("OPENBLAS_NUM_THREADS =", os.environ.get("OPENBLAS_NUM_THREADS"), flush=True)
print("PySCF lib.num_threads() =", lib.num_threads(), flush=True)


PBS = "6-311g**"
AUXBASIS = "weigend"
CHECKPOINT = "pdft_optimizer_trust_ridge1e35.pkl"
WARMSTART_PKL = "pdft_optimizer_trust_ridge1e34.pkl"
PHASE1_NSTEPS = 5
PHASE1_ALPHA = 1.0
PHASE1_MAX_STEP = 1.0
PHASE2_STEP_CAP_SCALE = 2.0
PHASE2_MAX_STEP_NORM_BASE = 0.03

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


# ---------------------------------------------------------------------------
# PBS / grid ↔ AO helpers
# ---------------------------------------------------------------------------

def vp_ao_from_pbs_coeffs(b, s3):
    """AO Vp from PBS coeffs: V_μν = Σ_t b_t S3_μνt (symmetrized)."""
    b = np.asarray(b, dtype=np.float64).ravel()
    s3 = np.asarray(s3, dtype=np.float64)
    if s3.ndim != 3:
        raise ValueError(f"S3 must be 3D, got shape {s3.shape}")
    if b.shape[0] != s3.shape[2]:
        raise ValueError(f"len(b)={b.size} != npbs={s3.shape[2]}")
    vp_ao = np.einsum("mnp,p->mn", s3, b, optimize=True)
    return 0.5 * (vp_ao + vp_ao.T)


def v2V(v_grid, w, phi):
    """Grid potential v[p] → AO matrix (same as 2.4capdft.py)."""
    ngrid = phi.shape[0]
    v_grid = np.asarray(v_grid, dtype=np.float64).ravel()
    v_ao = np.einsum("p,pu,pv->uv", w * v_grid, phi, phi, optimize=True)
    #v_ao /= ngrid
    return 0.5 * (v_ao + v_ao.T)


def vp_grid_from_pbs(b, phi_pbs):
    """PBS coeffs → grid values: v[p] = Σ_t b_t χ_t(r_p)."""
    return phi_pbs @ np.asarray(b, dtype=np.float64).ravel()


def density_from_dm_on_grid(phi, dm):
    """Real-space density n(r_p) from AO values and AO density matrix."""
    phi = np.asarray(phi, dtype=np.float64)
    dm = np.asarray(dm, dtype=np.float64)
    return np.einsum("pu,uv,pv->p", phi, dm, phi, optimize=True)


def project_grid_vp_to_pbs(vp_grid, phi_pbs):
    """Least-squares fit: phi_pbs @ b ≈ vp_grid (capdft grid → PBS coeffs)."""
    phi_pbs = np.asarray(phi_pbs, dtype=np.float64)
    vp_grid = np.asarray(vp_grid, dtype=np.float64).ravel()
    b, residuals, _, _ = np.linalg.lstsq(phi_pbs, vp_grid, rcond=None)
    if residuals.size:
        residual = float(np.sqrt(residuals[0]))
    else:
        residual = float(np.linalg.norm(phi_pbs @ b - vp_grid))
    return np.asarray(b, dtype=np.float64).ravel(), residual


# ---------------------------------------------------------------------------
# Fragment SCF
# ---------------------------------------------------------------------------

def collect_pdft_state(l, r):
    """Ensemble energy and spin DMs after SCF."""
    dal, dbl = l.get_D()
    dar, dbr = r.get_D()
    return l.get_E() + r.get_E(), dal, dbl, dar, dbr


def run_pdft_scf(l, r, vpl, vpr):
    """Run ensemble SCF with AO Vp matrices; return Ef and spin DMs."""
    t0 = time.time()
    l.scf(vpl)
    print(f"[timing] left/O ensemble SCF = {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    r.scf(vpr)
    print(f"[timing] right/Al ensemble SCF = {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    out = collect_pdft_state(l, r)
    print(f"[timing] collect state = {time.time() - t0:.1f}s", flush=True)
    return out


# ---------------------------------------------------------------------------
# System / grid / S3 setup
# ---------------------------------------------------------------------------

def build_pbs_mol(geo):
    """PBS basis functions evaluated on the full-system geometry."""
    pbs_mol = gto.Mole()
    pbs_mol.atom = geo.atom
    pbs_mol.unit = geo.unit
    pbs_mol.basis = PBS
    pbs_mol.charge = geo.charge
    pbs_mol.spin = geo.spin
    pbs_mol.build()
    return pbs_mol


def build_grid(geo, pbs_mol):
    """Level-3 DFT grid and AO/PBS values on grid points."""
    grid = dft.gen_grid.Grids(geo)
    grid.level = 3
    grid.build()
    coords = grid.coords
    w = grid.weights
    phi_geo = dft.numint.eval_ao(geo, coords, deriv=0)
    phi_pbs = dft.numint.eval_ao(pbs_mol, coords, deriv=0)
    return coords, w, phi_geo, phi_pbs


def compute_s3_tensors(sys, geo, pbs_mol):
    """S3 for left, right, and full AO spaces; PBS from full geo (same npbs)."""
    kwargs = dict(
        pbs=PBS,
        pbs_mol=pbs_mol,
        use_density_fit=True,
        auxbasis=AUXBASIS,
        ref=2,
    )
    s3l, _ = s3mod.compute_s3(sys["lgeo1"], basis=sys["lgeo1"].basis, **kwargs)
    s3r, _ = s3mod.compute_s3(sys["rgeo1"], basis=sys["rgeo1"].basis, **kwargs)
    s3, _ = s3mod.compute_s3(geo, basis=geo.basis, **kwargs)
    return s3l, s3r, s3


def build_system():
    with open("../2.7dftref/al2.7_sigma0.002_last_dm.pkl", "rb") as f:
        data_ref = pickle.load(f)
    daref = data_ref["Da_last_normal"]
    dbref = data_ref["Db_last_normal"]

    geo = gto.Mole()
    geo.atom = "4.xyz"
    geo.unit = "angstrom"
    geo.basis = {"O": OBASIS, "Al": ALBASIS}
    geo.charge = 0
    geo.spin = 2
    geo.build()

    w1o, w2o = 0.794146, 0.205854
    w1al, w2al = 0.897073, 0.102927

    lgeo1 = gto.Mole()
    lgeo1.atom = "o.xyz"
    lgeo1.unit = "angstrom"
    lgeo1.basis = {"O": OBASIS, "ghost-Al": "6-31g*"}
    lgeo1.spin = 2
    lgeo1.build()

    lgeo2 = gto.Mole()
    lgeo2.atom = "o.xyz"
    lgeo2.unit = "angstrom"
    lgeo2.basis = {"O": OBASIS, "ghost-Al": "6-31g*"}
    lgeo2.charge = -1
    lgeo2.spin = 1
    lgeo2.build()

    ldft1 = fragments.FragmentDFT(lgeo1, "pbe", newton=True)
    ldft2 = fragments.FragmentDFT(lgeo2, "pbe", newton=True)
    l = fragments.ens([ldft1, ldft2], [w1o, w2o])

    rgeo1 = gto.Mole()
    rgeo1.atom = "rgeo.xyz"
    rgeo1.unit = "angstrom"
    rgeo1.basis = {"ghost-O": "6-31g*", "Al": ALBASIS}
    rgeo1.spin = 0
    rgeo1.build()

    rgeo2 = gto.Mole()
    rgeo2.atom = "rgeo.xyz"
    rgeo2.unit = "angstrom"
    rgeo2.basis = {"ghost-O": "6-31g*", "Al": ALBASIS}
    rgeo2.charge = 2
    rgeo2.spin = 2
    rgeo2.build()

    rdft1 = fragments.FragmentDFT(rgeo1, "pbe", metal=True, smearing=True, newton=False)
    rdft2 = fragments.FragmentDFT(
        rgeo2, "pbe", metal=True, smearing=True, newton=False,sigma=0.005)
    r = fragments.ens([rdft1, rdft2], [w1al, w2al])

    print("left ensemble electron number", l.get_nelec())
    print("right ensemble electron number", r.get_nelec())

    return {
        "geo": geo,
        "lgeo1": lgeo1,
        "rgeo1": rgeo1,
        "l": l,
        "r": r,
        "rdft1": rdft1,
        "rdft2": rdft2,
        "daref": daref,
        "dbref": dbref,
    }


def atomic_pickle_save(path, data):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Warm-start from capdft / optimizer checkpoint
# ---------------------------------------------------------------------------

def fragment_D_from_pickle(dm):
    """Normalize capdft ``dm_ig`` (tuple or array) to FragmentDFT ``self.D`` format."""
    if isinstance(dm, (tuple, list)) and len(dm) == 2:
        da = np.asarray(dm[0], dtype=np.float64)
        db = np.asarray(dm[1], dtype=np.float64)
        return np.array((da, db))
    arr = np.asarray(dm, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[0] == 2:
        return arr
    raise ValueError(f"unexpected fragment DM format: {type(dm)}, shape {getattr(arr, 'shape', None)}")


def load_fragment_dm_guess(data, sys):
    """
    Load ``dm_ig1``/``dm_ig2`` onto ``rdft1``/``rdft2`` (same as ``2.4capdft.py``).

    Returns a copy kept for the first lagrangian SCF step.
    """
    if data is None:
        return {}

    guess = {}
    for pkl_key, frag_key in (("dm_ig1", "rdft1"), ("dm_ig2", "rdft2")):
        raw = data.get(pkl_key)
        if raw is None:
            continue
        d = fragment_D_from_pickle(raw)
        sys[frag_key].D = d
        guess[frag_key] = np.array(d, copy=True)
        print(f"[warm-start] {frag_key}.D ← pickle {pkl_key}")
    return guess


def apply_fragment_dm_guess(sys, fragment_dm_guess):
    """Set fragment ``D`` before SCF (used on lagrangian step 1)."""
    for frag_key, d in fragment_dm_guess.items():
        sys[frag_key].D = np.array(d, copy=True)


def load_warmstart_pickle(sys):
    """Prefer explicit warm-start pickle; fall back to optimizer checkpoint."""
    for path in (WARMSTART_PKL, CHECKPOINT):
        if os.path.isfile(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            print(f"[warm-start] loaded {path}")
            return data, path
    return None, None


def init_pbs_coeffs(data, npbs, phi_pbs, warmstart_path=None):
    """
    Initial PBS coefficient vector b0.

    - Optimizer restart: pickle ``vp`` already has length ``npbs``.
    - Capdft restart: pickle ``vp`` is grid-sized → project onto ``phi_pbs``.
    - Otherwise: b0 = 0.
    """
    if data is None:
        print("[init] no pickle; b0 = 0")
        return np.zeros(npbs, dtype=np.float64)

    if "vp" not in data:
        print("[init] no vp in pickle; b0 = 0")
        return np.zeros(npbs, dtype=np.float64)

    vp = np.asarray(data["vp"], dtype=np.float64).ravel()
    ngrid = phi_pbs.shape[0]

    if vp.size == npbs:
        src = warmstart_path if warmstart_path is not None else "loaded warm-start pickle"
        print(f"[init] b0 from {src}, npbs={npbs}")
        return vp

    if vp.size == ngrid:
        #b0, residual = project_grid_vp_to_pbs(vp, phi_pbs)
        #print(
        #    f"[init] b0 from capdft grid → PBS projection, "
        #    f"npbs={npbs}, residual={residual:.5e}"
        #)
        return np.zeros(npbs, dtype=np.float64)

    print(
        f"[init] vp length {vp.size} (expected npbs={npbs} or ngrid={ngrid}); "
        "b0 = 0"
    )
    return np.zeros(npbs, dtype=np.float64)


def lagrangian_initial_guess(sys, npbs, phi_pbs):
    """
    Warm-start for L-BFGS-B.

    Returns ``b0`` (PBS coeffs for ``x0``) and ``fragment_dm_guess`` for step-1 SCF.
    """
    data, warmstart_path = load_warmstart_pickle(sys)
    print(f"[warm-start] b0 priority source = {WARMSTART_PKL} (fallback: {CHECKPOINT})")
    fragment_dm_guess = load_fragment_dm_guess(data, sys)
    b0 = init_pbs_coeffs(data, npbs, phi_pbs, warmstart_path=warmstart_path)
    #b0 = np.zeros(npbs, dtype=np.float64)
    print(f"[warm-start] Lagrangian x0 ready, |b0| = {np.linalg.norm(b0):.5e}")
    return b0, fragment_dm_guess


# ---------------------------------------------------------------------------
# Post-optimization diagnostics
# ---------------------------------------------------------------------------

def vp_matrix_diff(a, b):
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    fnorm = float(np.linalg.norm(diff))
    ref = float(np.linalg.norm(b))
    rel = fnorm / ref if ref > 0 else float("nan")
    return fnorm, rel, float(np.max(np.abs(diff)))


def compare_vp_s3_vs_grid(b, s3l, s3r, s3, w, phi_l1, phi_r1, phi_geo, phi_pbs):
    """Compare AO Vp from S3 contraction vs grid v2V for converged PBS coeffs."""
    b = np.asarray(b, dtype=np.float64).ravel()
    v_grid = vp_grid_from_pbs(b, phi_pbs)

    cases = (
        ("Vpl (left)", s3l, phi_l1),
        ("Vpr (right)", s3r, phi_r1),
        ("Vpref (full)", s3, phi_geo),
    )
    print("\n=== Vp comparison: S3 vs grid (final b) ===")
    for label, s3_frag, phi_frag in cases:
        vp_s3 = vp_ao_from_pbs_coeffs(b, s3_frag)
        vp_grid_ao = v2V(v_grid, w, phi_frag)
        fnorm, rel, maxabs = vp_matrix_diff(vp_s3, vp_grid_ao)
        print(
            f"{label}: ||S3 - grid||_F = {fnorm:.6e}  "
            f"rel = {rel:.6e}  max|Δ| = {maxabs:.6e}"
        )


# ---------------------------------------------------------------------------
# Lagrangian evaluation and optimization
# ---------------------------------------------------------------------------

def eval_L_grad(b, l, r, dref, s3l, s3r, s3, sys):
    """Evaluate Lagrangian L(b) and gradient ∇L (for maximization)."""
    b = np.asarray(b, dtype=np.float64)
    vpl = vp_ao_from_pbs_coeffs(b, s3l)
    vpr = vp_ao_from_pbs_coeffs(b, s3r)
    vpref = vp_ao_from_pbs_coeffs(b, s3)

    ef, dal, dbl, dar, dbr = run_pdft_scf(l, r, vpl, vpr)
    lag = (
        ef
        + np.trace(dal @ vpl)
        + np.trace(dbl @ vpl)
        + np.trace(dar @ vpr)
        + np.trace(dbr @ vpr)
        - np.trace(dref @ vpref)
    )
    grad = (
        np.einsum("ij,ijt->t", dal + dbl, s3l)
        + np.einsum("ij,ijt->t", dar + dbr, s3r)
        - np.einsum("ij,ijt->t", dref, s3)
    )
    state = {
        "vp": b.copy(),  # key name kept for checkpoint compatibility
        "Vpl": vpl,
        "Vpr": vpr,
        "Vpref": vpref,
        "Dal": dal,
        "Dbl": dbl,
        "Dar": dar,
        "Dbr": dbr,
        "Dref": dref,
        "Ef": ef,
        "L": lag,
        "grad": grad,
        "dm_ig1": sys["rdft1"].get_rdm1(),
        "dm_ig2": sys["rdft2"].get_rdm1(),
    }
    return lag, grad, state


def save_lagrangian_checkpoint(state, iter_label):
    payload = dict(state)
    payload["iter"] = iter_label
    t0 = time.time()
    atomic_pickle_save(CHECKPOINT, payload)
    print(f"[timing] checkpoint save = {time.time() - t0:.1f}s", flush=True)


def run_phase1_damped_ascent(
    b0,
    l,
    r,
    dref,
    s3l,
    s3r,
    s3,
    sys,
    fragment_dm_guess=None,
    n_steps=PHASE1_NSTEPS,
    alpha=PHASE1_ALPHA,
    max_step=PHASE1_MAX_STEP,
):
    """Phase 1: damped gradient ascent — push b into a safe region for SCF."""
    b = np.asarray(b0, dtype=np.float64).copy()
    print(
        f"[phase1] damped ascent: {n_steps} steps, "
        f"alpha={alpha}, max_step={max_step}"
    )

    for k in range(n_steps):
        if k == 0 and fragment_dm_guess:
            apply_fragment_dm_guess(sys, fragment_dm_guess)
            print("[phase1] step 1 SCF: dm0 from dm_ig1/dm_ig2 on rdft1/rdft2")

        t0 = time.perf_counter()
        lag, grad, state = eval_L_grad(b, l, r, dref, s3l, s3r, s3, sys)
        elapsed = time.perf_counter() - t0

        db = alpha * grad
        db_norm = np.linalg.norm(db)
        if db_norm > 0:
            db *= min(1.0, max_step / db_norm)
        b = b + db

        state["vp"] = b.copy()
        state["elapsed_s"] = elapsed
        save_lagrangian_checkpoint(state, iter_label=f"phase1-{k + 1}")
        print(
            f"[phase1] step={k + 1}/{n_steps} L={lag:.8f} Ef={state['Ef']:.8f} "
            f"|grad|={np.linalg.norm(grad):.5e} |db|={np.linalg.norm(db):.5e} "
            f"t={elapsed:.1f}s"
        )
    return b


def make_lagrangian(l, r, dref, s3l, s3r, s3, sys):
    itera = 0

    def lagrangian(b):
        nonlocal itera
        itera += 1

        print(
            f"[EVAL step={itera}] |b|={np.linalg.norm(b):.6e} "
            f"max|b|={np.max(np.abs(b)):.6e}",
            flush=True,
        )

        t0 = time.perf_counter()

        t1 = time.time()
        vpl = vp_ao_from_pbs_coeffs(b, s3l)
        vpr = vp_ao_from_pbs_coeffs(b, s3r)
        vpref = vp_ao_from_pbs_coeffs(b, s3)
        print(f"[timing] build AO Vp from S3 = {time.time() - t1:.1f}s", flush=True)
        print(
            f"[diagnostic] |vpl|={np.linalg.norm(vpl):.6e} max|vpl|={np.max(np.abs(vpl)):.6e}",
            flush=True,
        )
        print(
            f"[diagnostic] |vpr|={np.linalg.norm(vpr):.6e} max|vpr|={np.max(np.abs(vpr)):.6e}",
            flush=True,
        )
        print(
            f"[diagnostic] |vpref|={np.linalg.norm(vpref):.6e} max|vpref|={np.max(np.abs(vpref)):.6e}",
            flush=True,
        )

        ef, dal, dbl, dar, dbr = run_pdft_scf(l, r, vpl, vpr)
        lag = (
            ef
            + np.trace(dal @ vpl)
            + np.trace(dbl @ vpl)
            + np.trace(dar @ vpr)
            + np.trace(dbr @ vpr)
            - np.trace(dref @ vpref)
        )

        t1 = time.time()
        grad = (
            np.einsum("ij,ijt->t", dal + dbl, s3l, optimize=True)
            + np.einsum("ij,ijt->t", dar + dbr, s3r, optimize=True)
            - np.einsum("ij,ijt->t", dref, s3, optimize=True)
        )
        print(f"[timing] grad contraction = {time.time() - t1:.1f}s", flush=True)

        nl = density_from_dm_on_grid(sys["phi_l1"], dal + dbl)
        nr = density_from_dm_on_grid(sys["phi_r1"], dar + dbr)
        nf = nl + nr
        nref = density_from_dm_on_grid(sys["phi_geo"], dref)
        print("Npdft", np.sum(nf * sys["w"]))
        l1 = float(np.sum(np.abs(nf - nref) * sys["w"]))

        state = {
            "vp": np.asarray(b, dtype=np.float64).copy(),  # key name kept for checkpoint compatibility
            "Vpl": vpl,
            "Vpr": vpr,
            "Vpref": vpref,
            "Dal": dal,
            "Dbl": dbl,
            "Dar": dar,
            "Dbr": dbr,
            "Dref": dref,
            "Ef": ef,
            "L": lag,
            "L1": l1,
            "grad": grad,
            "dm_ig1": sys["rdft1"].get_rdm1(),
            "dm_ig2": sys["rdft2"].get_rdm1(),
        }

        elapsed = time.perf_counter() - t0
        state["elapsed_s"] = elapsed
        sys["_last_eval_b"] = np.asarray(b, dtype=np.float64).copy()
        sys["_last_eval_state"] = state
        save_lagrangian_checkpoint(state, iter_label=itera)
        print(
            f"[EVAL step={itera:4d}] L={lag:.8f} Ef={state['Ef']:.8f} "
            f"L1={state['L1']:.8e} |grad|={np.linalg.norm(grad):.5e} t={elapsed:.1f}s"
        )
        return -lag, -grad

    return lagrangian


def main():
    t_main = time.perf_counter()

    t0 = time.perf_counter()
    sys = build_system()
    print(f"[setup] build_system t={time.perf_counter() - t0:.1f}s")

    geo = sys["geo"]
    l = sys["l"]
    r = sys["r"]
    dref = sys["daref"] + sys["dbref"]

    print("building grid (level 3)...")
    t0 = time.perf_counter()
    pbs_mol = build_pbs_mol(geo)
    coords, w, phi_geo, phi_pbs = build_grid(geo, pbs_mol)
    phi_l1 = dft.numint.eval_ao(sys["lgeo1"], coords, deriv=0)
    phi_r1 = dft.numint.eval_ao(sys["rgeo1"], coords, deriv=0)
    sys["w"] = w
    sys["phi_geo"] = phi_geo
    sys["phi_l1"] = phi_l1
    sys["phi_r1"] = phi_r1
    print(f"[setup] grid t={time.perf_counter() - t0:.1f}s")

    print("computing S3 tensors (DF)...")
    t0 = time.perf_counter()
    s3l, s3r, s3 = compute_s3_tensors(sys, geo, pbs_mol)
    print(
        f"S3 shapes: left={s3l.shape} right={s3r.shape} ref={s3.shape} "
        f"t={time.perf_counter() - t0:.1f}s"
    )

    t0 = time.perf_counter()
    b0, fragment_dm_guess = lagrangian_initial_guess(sys, s3.shape[2], phi_pbs)
    print(f"[setup] warm-start t={time.perf_counter() - t0:.1f}s")

    print("[main] skipping phase1; starting phase2 directly from b0")

    print("starting phase 2: L-BFGS-B optimization")
    print("starting phase 2: bounded L-BFGS-B outer loop")
    t0 = time.perf_counter()

    b = b0.copy()
    lagrangian = make_lagrangian(l, r, dref, s3l, s3r, s3, sys)

    target_grad = 1e-3
    max_outer = 20
    accepted_state = None
    last_L = None
    last_Ef = None
    next_step_scale = 1.0
    warned_missing_l1 = False

    for outer in range(max_outer):
        if last_Ef is None or last_Ef < -12264.0:
            max_step_norm = 0.10
        elif last_Ef < -12263.0:
            max_step_norm = 0.06
        else:
            max_step_norm = 0.03
        max_step_norm *= next_step_scale
        next_step_scale = 1.0
        delta = max_step_norm / np.sqrt(b.size)
        bounds = [(bi - delta, bi + delta) for bi in b]
        b_prev = b.copy()

        print(
            f"[phase2 outer {outer+1}] start "
            f"|b|={np.linalg.norm(b):.6e} "
            f"max|b|={np.max(np.abs(b)):.6e} "
            f"delta={delta:.3e}",
            flush=True,
        )

        b_old = b.copy()
        res = minimize(
            fun=lagrangian,
            x0=b,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 2,
                "maxls": 2,
                "maxcor": 3,
                "gtol": 1e-8,
                "ftol": 1e-14,
                },
        )

        b_new = res.x.copy()
        if "_last_eval_b" in sys and np.allclose(sys["_last_eval_b"], b_new, rtol=1e-10, atol=1e-12):
            accepted_state = sys["_last_eval_state"]
        else:
            print("[accepted eval] res.x was not the last evaluated point; evaluating accepted b", flush=True)
            _, _, accepted_state = eval_L_grad(b_new, l, r, dref, s3l, s3r, s3, sys)
        accepted_state["vp"] = b_new.copy()
        accepted_state["outer"] = outer + 1
        accepted_state["res_nfev"] = res.nfev
        accepted_state["res_nit"] = res.nit
        accepted_state["res_message"] = str(res.message)
        accepted_state["db_norm"] = np.linalg.norm(b_new - b_old)
        save_lagrangian_checkpoint(accepted_state, iter_label=f"phase2-outer-{outer+1}")

        raw_db = b_new - b_prev
        raw_step_norm = float(np.linalg.norm(raw_db))
        raw_max_abs_step = float(np.max(np.abs(raw_db)))
        line_search_alpha = getattr(res, "alpha", None)

        capped_db = raw_db.copy()
        if raw_step_norm > max_step_norm and raw_step_norm > 0.0:
            capped_db *= max_step_norm / raw_step_norm
            print(
                f"[phase2] step capped: raw_norm={raw_step_norm:.6e}, "
                f"capped_norm={np.linalg.norm(capped_db):.6e}",
                flush=True,
            )
        capped_step_norm = float(np.linalg.norm(capped_db))
        capped_max_abs_step = float(np.max(np.abs(capped_db)))
        b = b_prev + capped_db
        db_norm = capped_step_norm
        grad_norm = np.linalg.norm(res.jac)
        dL = accepted_state["L"] - last_L if last_L is not None else np.nan
        dEf = accepted_state["Ef"] - last_Ef if last_Ef is not None else np.nan
        last_L = accepted_state["L"]
        last_Ef = accepted_state["Ef"]
        print(f"[phase2 outer {outer+1}] |db| = {db_norm:.5e}")
        print(f"[phase2 outer {outer+1}] |grad| = {grad_norm:.5e}")
        print(f"[phase2 outer {outer+1}] last_Ef = {last_Ef}")
        print(f"[phase2 outer {outer+1}] adaptive max_step_norm = {max_step_norm:.6e}")
        print(f"[phase2 outer {outer+1}] delta = {delta:.6e}")
        print(f"[phase2 outer {outer+1}] raw_step_norm = {raw_step_norm:.6e}")
        print(f"[phase2 outer {outer+1}] capped_step_norm = {capped_step_norm:.6e}")
        print(f"[phase2 outer {outer+1}] dL = {dL}")
        print(f"[phase2 outer {outer+1}] dEf = {dEf}")
        print(f"[phase2 outer {outer+1}] |grad| = {grad_norm:.6e}")
        if line_search_alpha is None:
            print(
                f"[phase2 outer {outer+1}] raw_step_norm={raw_step_norm:.6e} "
                f"capped_step_norm={capped_step_norm:.6e} "
                f"raw_max_abs_step={raw_max_abs_step:.6e} "
                f"capped_max_abs_step={capped_max_abs_step:.6e} "
                f"step_cap={max_step_norm:.6e} line_search_alpha=n/a"
            )
        else:
            print(
                f"[phase2 outer {outer+1}] raw_step_norm={raw_step_norm:.6e} "
                f"capped_step_norm={capped_step_norm:.6e} "
                f"raw_max_abs_step={raw_max_abs_step:.6e} "
                f"capped_max_abs_step={capped_max_abs_step:.6e} "
                f"step_cap={max_step_norm:.6e} line_search_alpha={line_search_alpha:.6e}"
            )
        print(f"[phase2 outer {outer+1}] res.fun = {res.fun:.8f}")
        print(f"[phase2 outer {outer+1}] res.message = {res.message}")
        accepted_l1 = accepted_state.get("L1", float("nan"))
        if "L1" not in accepted_state and not warned_missing_l1:
            print(
                f"[WARNING] accepted_state missing L1 at outer {outer+1}; logging L1=nan",
                flush=True,
            )
            warned_missing_l1 = True
        print(
            f"[ACCEPTED outer {outer+1}] "
            f"L={accepted_state['L']:.8f} "
            f"Ef={accepted_state['Ef']:.8f} "
            f"L1={accepted_l1:.8e} "
            f"|grad|={np.linalg.norm(accepted_state['grad']):.5e} "
            f"|b|={np.linalg.norm(b_new):.6e} "
            f"max|b|={np.max(np.abs(b_new)):.6e} "
            f"|db|={accepted_state['db_norm']:.5e} "
            f"dL={dL:.8e} "
            f"dEf={dEf:.8e} "
            f"res.nfev={res.nfev} "
            f"res.nit={res.nit} "
            f"message={res.message}"
        )
        if dL < 0 or dEf < -1e-4:
            print("[WARNING] accepted objective decreased; consider reducing max_step_norm")

        real_grad_norm = grad_norm
        if real_grad_norm < target_grad:
            print(f"[phase2] reached target real |grad| < {target_grad:.1e}", flush=True)
            break
        print(f"[phase2 outer {outer+1}] finished message={res.message} |b|={np.linalg.norm(b):.6e}", flush=True)

    print(f"optimizer finished: {res.message} final L = {-res.fun:.8f}  niter = {res.nit}  success = {res.success}")
    b_final = accepted_state["vp"] if accepted_state is not None else b
    compare_vp_s3_vs_grid(b_final, s3l, s3r, s3, w, phi_l1, phi_r1, phi_geo, phi_pbs)
    print(f"[total] wall time t={time.perf_counter() - t_main:.1f}s")
    
if __name__ == "__main__":
    main()
