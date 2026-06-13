"""
Phase-2-only sign diagnostic for the 2.7 system.

Runs SciPy L-BFGS-B from four initial starts and logs:
- initial Ef, L, |grad|
- per-iteration Ef, L, |grad|, |db|, dEf, dL
- objective sign convention (L or -L)
- Jacobian sign convention (grad or -grad)

This script is diagnostic-only and does NOT write optimizer checkpoints.
"""

import argparse
import pickle
import time

import numpy as np
from scipy.optimize import minimize

from warmstart_compare_2_7 import (
    build_pbs_mol,
    build_system,
    collect_fragment_scf_status,
    compute_s3_tensors,
    eval_L_grad,
    get_b_from_ridge,
    load_fragment_dm_guess,
    load_ridge_b_map,
)


AO_TO_PBS_PKL = "ao_to_pbs_fit.pkl"
OPT_FINAL_PKL = "pdft_optimizer8.pkl"
OUT_LOG = "phase2_only_sign_test_2.7.log"
RIDGES = (1e-4, 1e-3)

# Current optimizer convention in pdftoptimizer.py:
#   SciPy minimizes fun, so we pass -L and -grad to maximize L.
SCIPY_OBJECTIVE_SIGN = -1.0  # +1 => objective=L, -1 => objective=-L
SCIPY_JAC_SIGN = -1.0        # +1 => jac=grad, -1 => jac=-grad


def log_line(fh, msg):
    print(msg, flush=True)
    fh.write(msg + "\n")
    fh.flush()


def evaluate_point(b, sys, s3l, s3r, s3, dref):
    t0 = time.perf_counter()
    lag, grad, state = eval_L_grad(b, sys["l"], sys["r"], dref, s3l, s3r, s3, sys)
    wall = time.perf_counter() - t0
    scf_status = collect_fragment_scf_status(sys)
    return {
        "Ef": float(state["Ef"]),
        "L": float(lag),
        "grad": np.asarray(grad, dtype=np.float64),
        "grad_norm": float(np.linalg.norm(grad)),
        "wall_s": float(wall),
        "scf_cycles": scf_status["total_cycles"],
        "scf_converged": scf_status["all_converged"],
    }


def sign_label(sign_value, positive_label, negative_label):
    if sign_value > 0:
        return positive_label
    if sign_value < 0:
        return negative_label
    raise ValueError("sign value must be non-zero")


def run_case(case_name, b0, s3l, s3r, s3, opt_data, log_fh, maxiter):
    sys = build_system()
    load_fragment_dm_guess(opt_data, sys)
    dref = sys["daref"] + sys["dbref"]

    b0 = np.asarray(b0, dtype=np.float64).copy()
    x_prev = b0.copy()
    eval_cache = {"x": None, "ev": None}
    iter_idx = 0

    def eval_cached(x):
        x = np.asarray(x, dtype=np.float64)
        if eval_cache["x"] is not None and np.array_equal(x, eval_cache["x"]):
            return eval_cache["ev"]
        ev = evaluate_point(x, sys, s3l, s3r, s3, dref)
        eval_cache["x"] = x.copy()
        eval_cache["ev"] = ev
        return ev

    def objective_and_jac(x):
        ev = eval_cached(x)
        obj = SCIPY_OBJECTIVE_SIGN * ev["L"]
        jac = SCIPY_JAC_SIGN * ev["grad"]
        return obj, jac

    initial = eval_cached(b0)
    log_line(log_fh, "")
    log_line(log_fh, f"===== case: {case_name} =====")
    log_line(
        log_fh,
        "[sign] "
        f"scipy objective = {sign_label(SCIPY_OBJECTIVE_SIGN, 'L', '-L')} ; "
        f"jac = {sign_label(SCIPY_JAC_SIGN, 'grad', '-grad')}",
    )
    log_line(
        log_fh,
        "[init] "
        f"Ef={initial['Ef']:.8f} L={initial['L']:.8f} |grad|={initial['grad_norm']:.6e} "
        f"cycles={initial['scf_cycles']} converged={initial['scf_converged']} "
        f"wall={initial['wall_s']:.1f}s",
    )

    def callback(xk):
        nonlocal iter_idx, x_prev
        iter_idx += 1
        xk = np.asarray(xk, dtype=np.float64)
        ev = eval_cached(xk)
        prev_ev = eval_cached(x_prev)
        db_norm = float(np.linalg.norm(xk - x_prev))
        dEf = float(ev["Ef"] - prev_ev["Ef"])
        dL = float(ev["L"] - prev_ev["L"])
        log_line(
            log_fh,
            f"[iter {iter_idx}] "
            f"Ef={ev['Ef']:.8f} L={ev['L']:.8f} |grad|={ev['grad_norm']:.6e} "
            f"|db|={db_norm:.6e} dEf={dEf:+.3e} dL={dL:+.3e} "
            f"cycles={ev['scf_cycles']} converged={ev['scf_converged']} "
            f"wall={ev['wall_s']:.1f}s",
        )
        x_prev = xk.copy()

    res = minimize(
        fun=objective_and_jac,
        x0=b0,
        method="L-BFGS-B",
        jac=True,
        callback=callback,
        options={
            "maxiter": int(maxiter),
            "maxls": 50,
            "maxcor": 3,
            "gtol": 1e-8,
            "ftol": 1e-14,
        },
    )

    x_final = np.asarray(res.x, dtype=np.float64)
    final_ev = eval_cached(x_final)
    log_line(
        log_fh,
        "[final] "
        f"message={res.message} success={res.success} nit={res.nit} "
        f"Ef={final_ev['Ef']:.8f} L={final_ev['L']:.8f} |grad|={final_ev['grad_norm']:.6e}",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Phase-2-only sign diagnostic for 2.7")
    parser.add_argument(
        "--maxiter",
        type=int,
        default=5,
        choices=(3, 5),
        help="L-BFGS maxiter (allowed: 3 or 5)",
    )
    parser.add_argument(
        "--out-log",
        default=OUT_LOG,
        help="Path to output log file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.perf_counter()

    ridge_map = load_ridge_b_map(AO_TO_PBS_PKL)
    with open(OPT_FINAL_PKL, "rb") as f:
        opt_data = pickle.load(f)

    b_opt_final = np.asarray(opt_data["vp"], dtype=np.float64).ravel()
    npbs = b_opt_final.size

    b_ridge_1e4 = get_b_from_ridge(ridge_map, RIDGES[0])
    b_ridge_1e3 = get_b_from_ridge(ridge_map, RIDGES[1])
    if b_ridge_1e4 is None or b_ridge_1e3 is None:
        raise RuntimeError("Missing ridge vectors for 1e-4 and/or 1e-3 in ao_to_pbs_fit.pkl")
    if b_ridge_1e4.size != npbs or b_ridge_1e3.size != npbs:
        raise RuntimeError("Ridge vector length mismatch vs b_opt_final")

    sys0 = build_system()
    geo = sys0["geo"]
    pbs_mol = build_pbs_mol(geo)
    s3l, s3r, s3 = compute_s3_tensors(sys0, geo, pbs_mol)
    if s3.shape[2] != npbs:
        raise RuntimeError(f"S3 npbs={s3.shape[2]} != len(b_opt_final)={npbs}")

    # User-requested order.
    cases = [
        ("b_zero", np.zeros(npbs, dtype=np.float64)),
        ("b_ridge_1e-3", b_ridge_1e3),
        ("b_ridge_1e-4", b_ridge_1e4),
        ("b_opt_final", b_opt_final),
    ]

    with open(args.out_log, "w") as log_fh:
        log_line(log_fh, "=== phase2-only sign test 2.7 ===")
        log_line(log_fh, f"AO_TO_PBS_PKL={AO_TO_PBS_PKL}")
        log_line(log_fh, f"OPT_FINAL_PKL={OPT_FINAL_PKL}")
        log_line(log_fh, f"maxiter={args.maxiter}")
        log_line(log_fh, f"S3 shapes left={s3l.shape}, right={s3r.shape}, full={s3.shape}")
        log_line(log_fh, "No checkpoint writes in this run.")

        for case_name, b_init in cases:
            run_case(
                case_name=case_name,
                b0=b_init,
                s3l=s3l,
                s3r=s3r,
                s3=s3,
                opt_data=opt_data,
                log_fh=log_fh,
                maxiter=args.maxiter,
            )

        log_line(log_fh, "")
        log_line(log_fh, f"[done] total wall time = {time.perf_counter() - t0:.1f}s")

    print(f"[save] wrote {args.out_log}", flush=True)


if __name__ == "__main__":
    main()
