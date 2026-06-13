"""
Short warm-start comparison for PBS optimizer initial b vectors.

This script does NOT run L-BFGS and does NOT overwrite optimizer checkpoints.
It evaluates a few candidate initial PBS coefficient vectors, then performs a
small number of damped-ascent probe steps to test smoothness.
"""

import time
import pickle

import numpy as np

from pdftoptimizer import (
    atomic_pickle_save,
    build_system,
    build_pbs_mol,
    compute_s3_tensors,
    eval_L_grad,
    load_fragment_dm_guess,
)


AO_TO_PBS_PKL = "ao_to_pbs_fit.pkl"
OPT_FINAL_PKL = "pdft_optimizer8.pkl"
OUT_LOG = "warmstart_compare_2.7.log"
OUT_PKL = "warmstart_compare_2.7.pkl"

RIDGES = (1e-4, 1e-3)
N_STEPS = 5
ALPHA = 1.0
MAX_STEP_NORM = 0.05
USE_OPT_DM_GUESS = True


def vec_cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def collect_fragment_scf_status(sys):
    entries = []
    total_cycles = 0
    has_cycles = False
    converged_flags = []

    for side_label, ens_obj in (("left", sys["l"]), ("right", sys["r"])):
        for idx, frag in enumerate(ens_obj.fragments):
            if frag is None:
                continue
            solver = getattr(frag, "dftsolver", None)
            if solver is None:
                continue
            cyc = getattr(solver, "cycles", None)
            conv = getattr(solver, "converged", None)
            if cyc is not None:
                has_cycles = True
                total_cycles += int(cyc)
            if conv is not None:
                converged_flags.append(bool(conv))
            entries.append(
                {
                    "side": side_label,
                    "fragment_index": int(idx),
                    "cycles": None if cyc is None else int(cyc),
                    "converged": None if conv is None else bool(conv),
                }
            )

    all_converged = all(converged_flags) if converged_flags else None
    return {
        "entries": entries,
        "total_cycles": int(total_cycles) if has_cycles else None,
        "all_converged": all_converged,
    }


def load_ridge_b_map(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    scan = data.get("fit_summary", {}).get("ridge_scan", [])
    ridge_map = {}
    for row in scan:
        ridge = float(row["ridge"])
        ridge_map[ridge] = np.asarray(row["vp"], dtype=np.float64).ravel()
    return ridge_map


def get_b_from_ridge(ridge_map, target):
    for ridge, b in ridge_map.items():
        if np.isclose(ridge, target, rtol=0.0, atol=1e-15):
            return np.array(b, copy=True)
    return None


def evaluate_once(name, b, b_init, b_opt_final, sys, s3l, s3r, s3, dref):
    t0 = time.perf_counter()
    lag, grad, state = eval_L_grad(b, sys["l"], sys["r"], dref, s3l, s3r, s3, sys)
    wall = time.perf_counter() - t0

    scf_status = collect_fragment_scf_status(sys)
    grad_norm = float(np.linalg.norm(grad))
    to_opt_from_init = b_opt_final - b_init
    grad_cos = vec_cosine(grad, to_opt_from_init)

    rec = {
        "name": name,
        "b_norm": float(np.linalg.norm(b)),
        "b_max_abs": float(np.max(np.abs(b))),
        "Ef": float(state["Ef"]),
        "L": float(lag),
        "grad_norm": grad_norm,
        "scf_total_cycles": scf_status["total_cycles"],
        "scf_converged": scf_status["all_converged"],
        "scf_detail": scf_status["entries"],
        "wall_time_s": float(wall),
        "cos_grad_to_bopt_minus_binit": grad_cos,
        "cos_grad_to_bopt_minus_bcurrent": vec_cosine(grad, b_opt_final - b),
        "grad": np.asarray(grad, dtype=np.float64),
    }
    return rec


def damped_step(grad, alpha, max_step_norm):
    db = alpha * np.asarray(grad, dtype=np.float64)
    db_norm = np.linalg.norm(db)
    if db_norm > 0.0:
        scale = min(1.0, max_step_norm / db_norm)
        db = db * scale
    return db


def log_line(fh, msg):
    print(msg)
    fh.write(msg + "\n")
    fh.flush()


def run_case(case_name, b_init, b_opt_final, s3l, s3r, s3, opt_data, log_fh):
    sys = build_system()
    if USE_OPT_DM_GUESS:
        load_fragment_dm_guess(opt_data, sys)
    dref = sys["daref"] + sys["dbref"]

    b = np.asarray(b_init, dtype=np.float64).copy()
    b0 = np.asarray(b_init, dtype=np.float64).copy()

    log_line(log_fh, "")
    log_line(log_fh, f"===== case: {case_name} =====")
    log_line(
        log_fh,
        f"[init] |b|={np.linalg.norm(b):.6e} max|b|={np.max(np.abs(b)):.6e}",
    )

    initial = evaluate_once(case_name, b, b0, b_opt_final, sys, s3l, s3r, s3, dref)
    log_line(
        log_fh,
        "[eval-0] "
        f"Ef={initial['Ef']:.8f} L={initial['L']:.8f} |grad|={initial['grad_norm']:.6e} "
        f"cycles={initial['scf_total_cycles']} converged={initial['scf_converged']} "
        f"wall={initial['wall_time_s']:.1f}s "
        f"cos(grad,b_opt-b_init)={initial['cos_grad_to_bopt_minus_binit']:.6f}",
    )

    steps = []
    prev_eval = initial
    grad = np.asarray(initial["grad"], dtype=np.float64)
    for k in range(1, N_STEPS + 1):
        db = damped_step(grad, alpha=ALPHA, max_step_norm=MAX_STEP_NORM)
        b = b + db

        ev = evaluate_once(case_name, b, b0, b_opt_final, sys, s3l, s3r, s3, dref)
        ev["step"] = int(k)
        ev["step_norm"] = float(np.linalg.norm(db))
        ev["L_delta"] = float(ev["L"] - prev_eval["L"])
        ev["Ef_delta"] = float(ev["Ef"] - prev_eval["Ef"])
        steps.append(ev)

        log_line(
            log_fh,
            f"[step {k}] "
            f"|db|={ev['step_norm']:.6e} "
            f"Ef={ev['Ef']:.8f} (d={ev['Ef_delta']:+.3e}) "
            f"L={ev['L']:.8f} (d={ev['L_delta']:+.3e}) "
            f"|grad|={ev['grad_norm']:.6e} "
            f"cycles={ev['scf_total_cycles']} converged={ev['scf_converged']} "
            f"wall={ev['wall_time_s']:.1f}s",
        )

        prev_eval = ev
        grad = np.asarray(ev["grad"], dtype=np.float64)

    grad_series = [initial["grad_norm"]] + [x["grad_norm"] for x in steps]
    smooth_summary = {
        "grad_nonincreasing": bool(
            all(grad_series[i + 1] <= grad_series[i] + 1e-12 for i in range(len(grad_series) - 1))
        ),
        "L_monotonic_increase": bool(all(x["L_delta"] >= -1e-12 for x in steps)),
        "Ef_monotonic_increase": bool(all(x["Ef_delta"] >= -1e-12 for x in steps)),
        "grad_series": [float(x) for x in grad_series],
    }
    log_line(
        log_fh,
        "[summary] "
        f"grad_nonincreasing={smooth_summary['grad_nonincreasing']} "
        f"L_monotonic_increase={smooth_summary['L_monotonic_increase']} "
        f"Ef_monotonic_increase={smooth_summary['Ef_monotonic_increase']}",
    )

    for rec in [initial] + steps:
        rec.pop("grad", None)

    return {
        "case_name": case_name,
        "initial": initial,
        "steps": steps,
        "smooth_summary": smooth_summary,
    }


def main():
    t_all = time.perf_counter()
    ridge_map = load_ridge_b_map(AO_TO_PBS_PKL)
    with open(OPT_FINAL_PKL, "rb") as f:
        opt_data = pickle.load(f)

    b_opt_final = np.asarray(opt_data["vp"], dtype=np.float64).ravel()
    npbs = b_opt_final.size

    b_ridge_1e4 = get_b_from_ridge(ridge_map, RIDGES[0])
    b_ridge_1e3 = get_b_from_ridge(ridge_map, RIDGES[1])
    if b_ridge_1e4 is None or b_ridge_1e3 is None:
        raise RuntimeError(
            "Missing ridge vectors in ao_to_pbs_fit.pkl for 1e-4 and/or 1e-3."
        )
    if b_ridge_1e4.size != npbs or b_ridge_1e3.size != npbs:
        raise RuntimeError("Ridge vector length mismatch against b_opt_final.")

    sys0 = build_system()
    geo = sys0["geo"]
    pbs_mol = build_pbs_mol(geo)
    s3l, s3r, s3 = compute_s3_tensors(sys0, geo, pbs_mol)
    if s3.shape[2] != npbs:
        raise RuntimeError(f"S3 npbs={s3.shape[2]} != len(b_opt_final)={npbs}")

    cases = [
        ("b_zero", np.zeros(npbs, dtype=np.float64)),
        ("b_ridge_1e-4", b_ridge_1e4),
        ("b_ridge_1e-3", b_ridge_1e3),
        ("b_opt_final", b_opt_final),
    ]

    with open(OUT_LOG, "w") as log_fh:
        log_line(log_fh, "=== warmstart compare 2.7 ===")
        log_line(log_fh, f"AO_TO_PBS_PKL={AO_TO_PBS_PKL}")
        log_line(log_fh, f"OPT_FINAL_PKL={OPT_FINAL_PKL}")
        log_line(log_fh, f"N_STEPS={N_STEPS} ALPHA={ALPHA} MAX_STEP_NORM={MAX_STEP_NORM}")
        log_line(log_fh, f"USE_OPT_DM_GUESS={USE_OPT_DM_GUESS}")
        log_line(log_fh, f"S3 shapes left={s3l.shape}, right={s3r.shape}, full={s3.shape}")

        results = []
        for case_name, b_init in cases:
            res = run_case(
                case_name=case_name,
                b_init=b_init,
                b_opt_final=b_opt_final,
                s3l=s3l,
                s3r=s3r,
                s3=s3,
                opt_data=opt_data,
                log_fh=log_fh,
            )
            results.append(res)

        total_wall = time.perf_counter() - t_all
        log_line(log_fh, "")
        log_line(log_fh, f"[done] total wall time = {total_wall:.1f}s")

    out = {
        "settings": {
            "AO_TO_PBS_PKL": AO_TO_PBS_PKL,
            "OPT_FINAL_PKL": OPT_FINAL_PKL,
            "RIDGES": list(RIDGES),
            "N_STEPS": int(N_STEPS),
            "ALPHA": float(ALPHA),
            "MAX_STEP_NORM": float(MAX_STEP_NORM),
            "USE_OPT_DM_GUESS": bool(USE_OPT_DM_GUESS),
        },
        "results": results,
        "total_wall_time_s": float(time.perf_counter() - t_all),
    }
    atomic_pickle_save(OUT_PKL, out)
    print(f"[save] wrote {OUT_LOG}")
    print(f"[save] wrote {OUT_PKL}")


if __name__ == "__main__":
    main()
