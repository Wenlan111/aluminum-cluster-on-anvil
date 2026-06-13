"""
Trust-krylov PDFT optimizer pilot for 2.7 A setup.

This script reuses the same system/geometry/fragment setup and Lagrangian
math from pdftoptimizer.py, but replaces the unconstrained L-BFGS-B run with
a small trust-krylov run using an analytic independent-particle response
Hessian in the PBS coefficient space.
"""

import time

import numpy as np
from scipy.optimize import minimize

import pdftoptimizer as opt


WARMSTART_PKL = "pdft_optimizer_trust_ridge1e34.pkl"
TARGET_GRAD = 1e-3
MAXITER = 5
GTOL = 3e-4
INITIAL_TRUST_RADIUS = 0.03
MAX_TRUST_RADIUS = 0.10
ETA = 0.15
HESSIAN_CHUNK = 8
OCC_TOL = 1e-8
DENOM_TOL = 1e-8
STABLE_GRAD_MAX = 1e-1
STABLE_NPDFT_TOL = 1e-2
STABLE_B_NORM_MAX = 6.0
STABLE_B_MAXABS_MAX = 1.5


def pdft_response_hessian(l, r, s3l, s3r, occ_tol=OCC_TOL,
                          denom_tol=DENOM_TOL, chunk=HESSIAN_CHUNK):
    """Independent-particle response Hessian d(grad_L)/db in PBS space."""
    npbs = s3l.shape[2]
    H_total = np.zeros((npbs, npbs), dtype=np.float64)

    def spin_hessian(C, eps, occ, s3):
        C = np.asarray(C, dtype=np.float64)
        eps = np.asarray(eps, dtype=np.float64)
        occ = np.asarray(occ, dtype=np.float64)
        max_occ = float(np.max(occ)) if occ.size else 0.0
        occ_idx = np.where(occ > occ_tol)[0]
        vir_idx = np.where(occ < max_occ - occ_tol)[0]
        H = np.zeros((npbs, npbs), dtype=np.float64)

        if occ_idx.size == 0 or vir_idx.size == 0:
            return H

        Cocc = C[:, occ_idx]
        eps_occ = eps[occ_idx]
        occ_occ = occ[occ_idx]

        for a0 in range(0, vir_idx.size, chunk):
            ids = vir_idx[a0:a0 + chunk]
            Cvir = C[:, ids]
            eps_vir = eps[ids]
            occ_vir = occ[ids]

            C3 = np.einsum("mi,na,mnt->iat", Cocc, Cvir, s3, optimize=True)
            denom = eps_occ[:, None] - eps_vir[None, :]
            occ_diff = occ_occ[:, None] - occ_vir[None, :]

            pair_factor = np.zeros_like(denom)
            mask = (np.abs(denom) > denom_tol) & (occ_diff > occ_tol)
            pair_factor[mask] = occ_diff[mask] / denom[mask]

            H += np.einsum("iau,iat,ia->ut", C3, C3, pair_factor, optimize=True)

        return H

    def fragment_hessian(frag, s3):
        mf = frag.dftsolver
        C = mf.mo_coeff
        eps = mf.mo_energy
        occ = mf.mo_occ

        C_arr = np.asarray(C)
        if isinstance(C, (tuple, list)) or (C_arr.ndim == 3 and C_arr.shape[0] == 2):
            return (
                spin_hessian(C[0], eps[0], occ[0], s3)
                + spin_hessian(C[1], eps[1], occ[1], s3)
            )
        return spin_hessian(C, eps, occ, s3)

    for ens_obj, s3 in ((l, s3l), (r, s3r)):
        for ens_weight, frag in zip(ens_obj.omega, ens_obj.fragments):
            if frag is None:
                continue
            if isinstance(ens_weight, (tuple, list)):
                ens_weight = sum(ens_weight)
            H_total += float(ens_weight) * fragment_hessian(frag, s3)

    return 0.5 * (H_total + H_total.T)


def is_stable_state(state):
    vals = [
        state.get("L", np.nan),
        state.get("Ef", np.nan),
        state.get("L1", np.nan),
        state.get("Npdft", np.nan),
    ]
    if not np.all(np.isfinite(vals)):
        return False
    grad_norm = float(np.linalg.norm(state["grad"]))
    b = np.asarray(state["vp"], dtype=np.float64)
    return (
        grad_norm <= STABLE_GRAD_MAX
        and abs(float(state["Npdft"]) - 666.0) <= STABLE_NPDFT_TOL
        and np.linalg.norm(b) <= STABLE_B_NORM_MAX
        and np.max(np.abs(b)) <= STABLE_B_MAXABS_MAX
    )


def main():
    t_main = time.perf_counter()
    print("=== trust-krylov scipy.minimize run (analytic response Hessian) ===")

    # Force warm-start source for this run.
    opt.WARMSTART_PKL = WARMSTART_PKL
    print(f"[config] WARMSTART_PKL={opt.WARMSTART_PKL}")
    print(f"[config] CHECKPOINT fallback={opt.CHECKPOINT}")
    print(f"[config] method=trust-krylov maxiter={MAXITER} gtol={GTOL:.1e}")
    print(
        f"[config] trust radii initial={INITIAL_TRUST_RADIUS:.3e} "
        f"max={MAX_TRUST_RADIUS:.3e} eta={ETA:.2f}"
    )
    print(f"[config] hessian chunk={HESSIAN_CHUNK}")

    t0 = time.perf_counter()
    sys = opt.build_system()
    print(f"[setup] build_system t={time.perf_counter() - t0:.1f}s")

    geo = sys["geo"]
    l = sys["l"]
    r = sys["r"]
    dref = sys["daref"] + sys["dbref"]

    print("building grid (level 3)...")
    t0 = time.perf_counter()
    pbs_mol = opt.build_pbs_mol(geo)
    coords, w, phi_geo, phi_pbs = opt.build_grid(geo, pbs_mol)
    phi_l1 = opt.dft.numint.eval_ao(sys["lgeo1"], coords, deriv=0)
    phi_r1 = opt.dft.numint.eval_ao(sys["rgeo1"], coords, deriv=0)
    sys["w"] = w
    sys["phi_geo"] = phi_geo
    sys["phi_l1"] = phi_l1
    sys["phi_r1"] = phi_r1
    print(f"[setup] grid t={time.perf_counter() - t0:.1f}s")

    print("computing S3 tensors (DF)...")
    t0 = time.perf_counter()
    s3l, s3r, s3 = opt.compute_s3_tensors(sys, geo, pbs_mol)
    print(
        f"S3 shapes: left={s3l.shape} right={s3r.shape} ref={s3.shape} "
        f"t={time.perf_counter() - t0:.1f}s"
    )

    t0 = time.perf_counter()
    b0, fragment_dm_guess = opt.lagrangian_initial_guess(sys, s3.shape[2], phi_pbs)
    print(f"[setup] warm-start t={time.perf_counter() - t0:.1f}s")

    if fragment_dm_guess:
        opt.apply_fragment_dm_guess(sys, fragment_dm_guess)
        print("[setup] applied dm_ig warm-start to rdft1/rdft2 before first evaluation")

    eval_count = 0
    last_state = None
    eval_history = []
    prev_lag = None
    prev_grad_norm = None
    cache_x = None
    cache_fg = None
    cache_H_x = None
    cache_H = None

    def same_x(a, b):
        return (
            a is not None
            and b is not None
            and np.allclose(a, b, rtol=1e-12, atol=1e-12)
        )

    def objective(b):
        nonlocal eval_count, last_state, prev_lag, prev_grad_norm, cache_x, cache_fg
        if same_x(cache_x, b):
            return cache_fg

        eval_count += 1
        t_eval = time.perf_counter()
        lag, grad, state = opt.eval_L_grad(b, l, r, dref, s3l, s3r, s3, sys)

        nl = opt.density_from_dm_on_grid(sys["phi_l1"], state["Dal"] + state["Dbl"])
        nr = opt.density_from_dm_on_grid(sys["phi_r1"], state["Dar"] + state["Dbr"])
        nf = nl + nr
        nref = opt.density_from_dm_on_grid(sys["phi_geo"], dref)
        l1 = float(np.sum(np.abs(nf - nref) * sys["w"]))
        npdft = float(np.sum(nf * sys["w"]))

        state["L1"] = l1
        state["Npdft"] = npdft
        state["elapsed_s"] = float(time.perf_counter() - t_eval)
        state["eval"] = eval_count
        state["vp"] = np.asarray(b, dtype=np.float64).copy()
        last_state = state
        eval_history.append(state)
        grad_norm = float(np.linalg.norm(grad))
        dL = lag - prev_lag if prev_lag is not None else np.nan
        dgrad = grad_norm - prev_grad_norm if prev_grad_norm is not None else np.nan

        opt.atomic_pickle_save(f"pdft_eval_{eval_count:04d}.pkl", state)
        opt.save_lagrangian_checkpoint(state, iter_label=f"trust-krylov-eval-{eval_count}")

        print(
            f"[EVAL {eval_count:4d}] "
            f"L={lag:.8f} Ef={state['Ef']:.8f} L1={l1:.8e} Npdft={npdft:.6f} "
            f"|grad|={grad_norm:.5e} "
            f"dL={dL:+.3e} d|grad|={dgrad:+.3e} "
            f"|b|={np.linalg.norm(b):.6e} max|b|={np.max(np.abs(b)):.6e} "
            f"t={state['elapsed_s']:.1f}s"
        )
        prev_lag = lag
        prev_grad_norm = grad_norm
        cache_x = np.asarray(b, dtype=np.float64).copy()
        cache_fg = (-lag, -grad)
        return cache_fg

    def hessp(b, p):
        nonlocal cache_H_x, cache_H
        objective(b)
        if not same_x(cache_H_x, b):
            t_hess = time.perf_counter()
            H_L = pdft_response_hessian(l, r, s3l, s3r)
            cache_H = -H_L
            cache_H_x = np.asarray(b, dtype=np.float64).copy()
            eigs = np.linalg.eigvalsh(cache_H)
            eig_min = float(eigs[0])
            eig_max = float(eigs[-1])
            print(
                f"[HESS] built H_scipy ||H||={np.linalg.norm(cache_H):.6e} "
                f"max|H|={np.max(np.abs(cache_H)):.6e} "
                f"eig_min={eig_min:.6e} eig_max={eig_max:.6e} "
                f"t={time.perf_counter() - t_hess:.1f}s",
                flush=True,
            )
        return cache_H @ np.asarray(p, dtype=np.float64)

    print("starting scipy.optimize.minimize (trust-krylov, analytic hessp)...")
    t0 = time.perf_counter()
    res = minimize(
        fun=objective,
        x0=b0,
        jac=True,
        hessp=hessp,
        method="trust-krylov",
        options={
            "maxiter": MAXITER,
            "gtol": GTOL,
            "initial_trust_radius": INITIAL_TRUST_RADIUS,
            "max_trust_radius": MAX_TRUST_RADIUS,
            "eta": ETA,
            "disp": True,
        },
    )
    print(f"[timing] minimize wall={time.perf_counter() - t0:.1f}s")

    stable_states = [state for state in eval_history if is_stable_state(state)]
    if stable_states:
        state_final = min(stable_states, key=lambda state: (state["L1"], -state["L"]))
        print(
            f"[select] selected stable eval {state_final['eval']} by lowest L1",
            flush=True,
        )
    elif eval_history:
        state_final = min(eval_history, key=lambda state: state["L1"])
        print(
            f"[select] no stable states; selected eval {state_final['eval']} by lowest L1",
            flush=True,
        )
    else:
        raise RuntimeError("No objective evaluations were recorded.")

    b_final = np.asarray(state_final["vp"], dtype=np.float64).copy()
    lag_final = float(state_final["L"])
    grad_final = np.asarray(state_final["grad"], dtype=np.float64)
    grad_norm_final = float(np.linalg.norm(grad_final))
    l1_final = float(state_final["L1"])
    npdft_final = float(state_final["Npdft"])

    print("\n=== final summary ===")
    print(f"success={res.success} status={res.status}")
    print(f"message={res.message}")
    print(
        f"nfev={getattr(res, 'nfev', 'n/a')} njev={getattr(res, 'njev', 'n/a')} "
        f"nhev={getattr(res, 'nhev', 'n/a')} nit={res.nit}"
    )
    print(f"final L={lag_final:.8f} Ef={state_final['Ef']:.8f} L1={l1_final:.8e}")
    print(
        f"final |grad|={grad_norm_final:.5e} Npdft={npdft_final:.6f} "
        f"|b|={np.linalg.norm(b_final):.6e} max|b|={np.max(np.abs(b_final)):.6e}"
    )
    if grad_norm_final < TARGET_GRAD:
        print(f"[stop] reached target |grad| < {TARGET_GRAD:.1e}")

    opt.compare_vp_s3_vs_grid(
        b_final, s3l, s3r, s3, w, phi_l1, phi_r1, phi_geo, phi_pbs
    )
    opt.atomic_pickle_save("pdft_trust_krylov_selected.pkl", state_final)
    print(f"[total] wall time t={time.perf_counter() - t_main:.1f}s")


if __name__ == "__main__":
    main()
