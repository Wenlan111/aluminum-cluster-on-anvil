"""
Lambda scan for selecting a safe initial PBS coefficient vector b0.

This script evaluates fixed partition potentials b = lambda * b_ridge and
collects SCF + density diagnostics. It does not run scipy.optimize.minimize and
does not update checkpoints.
"""

import csv
import json
import pickle
import time

import numpy as np
from pyscf import dft

from pdftoptimizer import (
    atomic_pickle_save,
    build_grid,
    build_pbs_mol,
    build_system,
    compute_s3_tensors,
    density_from_dm_on_grid,
    run_pdft_scf,
    vp_ao_from_pbs_coeffs,
)


AO_TO_PBS_PKL = "ao_to_pbs_fit.pkl"
OUT_CSV = "b0_lambda_scan.csv"
OUT_PKL = "b0_lambda_scan.pkl"
LAMBDAS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
TARGET_RIDGE = 1e-3

NPDFT_TARGET = 666.0
NPDFT_DEV_WARN = 0.5
GRAD_NORM_WARN = 0.1
CYCLE_NEAR_MAX_RATIO = 0.9
ABRUPT_DELTA_RATIO = 3.0
ABRUPT_GROWTH_RATIO = 1.5


def load_b_ridge(path, target_ridge):
    with open(path, "rb") as f:
        data = pickle.load(f)

    scan = data.get("fit_summary", {}).get("ridge_scan", [])
    if not scan:
        raise RuntimeError(
            f"Missing fit_summary.ridge_scan in {path}; cannot select ridge ~ {target_ridge}."
        )

    hits = []
    for row in scan:
        ridge = float(row.get("ridge", np.nan))
        if np.isfinite(ridge) and np.isclose(ridge, target_ridge, rtol=1e-6, atol=0.0):
            hits.append(row)

    if not hits:
        available = sorted(
            float(row.get("ridge")) for row in scan if np.isfinite(float(row.get("ridge", np.nan)))
        )
        raise RuntimeError(
            f"Could not find ridge ~ {target_ridge} in {path}. "
            f"Available ridge values: {available}"
        )

    row = hits[0]
    if "vp" not in row:
        raise RuntimeError(
            f"Ridge entry for {target_ridge} in {path} has no 'vp' vector."
        )
    b_ridge = np.asarray(row["vp"], dtype=np.float64).ravel()
    return b_ridge, float(row["ridge"])


def get_solver_e_tot(solver):
    val = getattr(solver, "e_tot", None)
    if val is not None:
        return float(val)
    mf = getattr(solver, "mf", None)
    val = getattr(mf, "e_tot", None)
    return None if val is None else float(val)


def get_solver_max_cycle(solver):
    val = getattr(solver, "max_cycle", None)
    if val is not None:
        return int(val)
    mf = getattr(solver, "mf", None)
    val = getattr(mf, "max_cycle", None)
    return None if val is None else int(val)


def collect_fragment_solver_diagnostics(sys):
    out = []
    for side_label, ens_obj in (("left", sys["l"]), ("right", sys["r"])):
        for idx, frag in enumerate(ens_obj.fragments):
            if frag is None:
                continue
            solver = getattr(frag, "dftsolver", None)
            if solver is None:
                continue

            converged = getattr(solver, "converged", None)
            last_cycle = getattr(solver, "_last_normal_cycle", None)
            max_cycle = get_solver_max_cycle(solver)
            e_tot = get_solver_e_tot(solver)

            out.append(
                {
                    "side": side_label,
                    "fragment_index": int(idx),
                    "converged": None if converged is None else bool(converged),
                    "_last_normal_cycle": None if last_cycle is None else int(last_cycle),
                    "max_cycle": max_cycle,
                    "e_tot": e_tot,
                }
            )
    return out


def evaluate_fixed_b(b, sys, s3l, s3r, s3, dref):
    b = np.asarray(b, dtype=np.float64).ravel()

    vpl = vp_ao_from_pbs_coeffs(b, s3l)
    vpr = vp_ao_from_pbs_coeffs(b, s3r)
    vpref = vp_ao_from_pbs_coeffs(b, s3)

    t0 = time.perf_counter()
    ef, dal, dbl, dar, dbr = run_pdft_scf(sys["l"], sys["r"], vpl, vpr)
    scf_wall_time_s = time.perf_counter() - t0

    lag = (
        ef
        + np.trace(dal @ vpl)
        + np.trace(dbl @ vpl)
        + np.trace(dar @ vpr)
        + np.trace(dbr @ vpr)
        - np.trace(dref @ vpref)
    )
    grad = (
        np.einsum("ij,ijt->t", dal + dbl, s3l, optimize=True)
        + np.einsum("ij,ijt->t", dar + dbr, s3r, optimize=True)
        - np.einsum("ij,ijt->t", dref, s3, optimize=True)
    )
    grad_norm = float(np.linalg.norm(grad))

    nl = density_from_dm_on_grid(sys["phi_l1"], dal + dbl)
    nr = density_from_dm_on_grid(sys["phi_r1"], dar + dbr)
    nf = nl + nr
    nref = density_from_dm_on_grid(sys["phi_geo"], dref)
    l1 = float(np.sum(np.abs(nf - nref) * sys["w"]))
    npdft = float(np.sum(nf * sys["w"]))

    solver_diag = collect_fragment_solver_diagnostics(sys)

    row = {
        "L": float(lag),
        "Ef": float(ef),
        "grad_norm": grad_norm,
        "L1": l1,
        "Npdft": npdft,
        "b_norm": float(np.linalg.norm(b)),
        "b_max_abs": float(np.max(np.abs(b))),
        "Vpl_norm": float(np.linalg.norm(vpl)),
        "Vpr_norm": float(np.linalg.norm(vpr)),
        "Vpref_norm": float(np.linalg.norm(vpref)),
        "Vpl_max_abs": float(np.max(np.abs(vpl))),
        "Vpr_max_abs": float(np.max(np.abs(vpr))),
        "Vpref_max_abs": float(np.max(np.abs(vpref))),
        "scf_wall_time_s": float(scf_wall_time_s),
        "solver_diagnostics": solver_diag,
    }
    return row


def flag_row_stability(row):
    solver_diag = row["solver_diagnostics"]
    flag_nonconverged = any(d["converged"] is False for d in solver_diag)
    flag_cycles_near_max = any(
        (d["_last_normal_cycle"] is not None and d["max_cycle"] is not None)
        and (d["_last_normal_cycle"] >= CYCLE_NEAR_MAX_RATIO * d["max_cycle"])
        for d in solver_diag
    )
    flag_npdft = abs(row["Npdft"] - NPDFT_TARGET) > NPDFT_DEV_WARN
    flag_grad = row["grad_norm"] > GRAD_NORM_WARN
    row["flag_nonconverged"] = bool(flag_nonconverged)
    row["flag_cycles_near_max"] = bool(flag_cycles_near_max)
    row["flag_npdft_deviation"] = bool(flag_npdft)
    row["flag_grad_norm_high"] = bool(flag_grad)


def mark_abrupt_by_deltas(rows, key, flag_key):
    if len(rows) < 3:
        return
    deltas = [rows[i][key] - rows[i - 1][key] for i in range(1, len(rows))]
    eps = 1e-12
    for i in range(2, len(rows)):
        d_prev = abs(deltas[i - 2])
        d_cur = abs(deltas[i - 1])
        if d_cur > ABRUPT_DELTA_RATIO * max(d_prev, eps):
            rows[i][flag_key] = True


def mark_abrupt_growth(rows, key, flag_key):
    eps = 1e-12
    for i in range(1, len(rows)):
        prev = abs(rows[i - 1][key])
        cur = abs(rows[i][key])
        if cur > ABRUPT_GROWTH_RATIO * max(prev, eps):
            rows[i][flag_key] = True


def annotate_neighbor_flags(rows):
    for row in rows:
        row["flag_nonsmooth_L"] = False
        row["flag_nonsmooth_Ef"] = False
        row["flag_nonsmooth_L1"] = False
        row["flag_abrupt_potential"] = False

    mark_abrupt_by_deltas(rows, "L", "flag_nonsmooth_L")
    mark_abrupt_by_deltas(rows, "Ef", "flag_nonsmooth_Ef")
    mark_abrupt_by_deltas(rows, "L1", "flag_nonsmooth_L1")

    potential_keys = [
        "b_norm",
        "b_max_abs",
        "Vpl_norm",
        "Vpr_norm",
        "Vpref_norm",
        "Vpl_max_abs",
        "Vpr_max_abs",
        "Vpref_max_abs",
    ]
    for key in potential_keys:
        tmp_flag = f"_tmp_abrupt_{key}"
        for row in rows:
            row[tmp_flag] = False
        mark_abrupt_growth(rows, key, tmp_flag)
        for row in rows:
            row["flag_abrupt_potential"] = row["flag_abrupt_potential"] or row[tmp_flag]
            del row[tmp_flag]


def any_flagged(row):
    return any(
        row[k]
        for k in (
            "flag_nonconverged",
            "flag_cycles_near_max",
            "flag_npdft_deviation",
            "flag_grad_norm_high",
            "flag_nonsmooth_L",
            "flag_nonsmooth_Ef",
            "flag_nonsmooth_L1",
            "flag_abrupt_potential",
        )
    )


def write_csv(path, rows):
    fields = [
        "lambda",
        "L",
        "Ef",
        "L1",
        "Npdft",
        "grad_norm",
        "b_norm",
        "b_max_abs",
        "Vpl_norm",
        "Vpr_norm",
        "Vpref_norm",
        "Vpl_max_abs",
        "Vpr_max_abs",
        "Vpref_max_abs",
        "scf_wall_time_s",
        "flag_nonconverged",
        "flag_cycles_near_max",
        "flag_npdft_deviation",
        "flag_grad_norm_high",
        "flag_nonsmooth_L",
        "flag_nonsmooth_Ef",
        "flag_nonsmooth_L1",
        "flag_abrupt_potential",
        "flag_any",
        "solver_diagnostics_json",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            csv_row = {k: row.get(k) for k in fields}
            csv_row["solver_diagnostics_json"] = json.dumps(
                row["solver_diagnostics"], sort_keys=True
            )
            w.writerow(csv_row)


def print_summary_table(rows):
    print("\n=== Lambda scan summary ===")
    print(
        "lambda   L            Ef           L1         Npdft      |grad|     "
        "SCF(s)  flags"
    )
    print("-" * 96)
    for row in rows:
        flags = []
        if row["flag_nonconverged"]:
            flags.append("nonconv")
        if row["flag_cycles_near_max"]:
            flags.append("nearmax")
        if row["flag_npdft_deviation"]:
            flags.append("Ndev")
        if row["flag_grad_norm_high"]:
            flags.append("grad")
        if row["flag_nonsmooth_L"] or row["flag_nonsmooth_Ef"] or row["flag_nonsmooth_L1"]:
            flags.append("nonsmooth")
        if row["flag_abrupt_potential"]:
            flags.append("Vpjump")
        flag_str = ",".join(flags) if flags else "ok"
        print(
            f"{row['lambda']:>5.2f}  {row['L']:> .8f}  {row['Ef']:> .8f}  "
            f"{row['L1']:> .3e}  {row['Npdft']:>8.3f}  {row['grad_norm']:> .3e}  "
            f"{row['scf_wall_time_s']:>6.1f}  {flag_str}"
        )


def main():
    print("=== b0 lambda scan (fixed-vp SCF only; no minimize) ===")
    print(f"[config] AO_TO_PBS_PKL={AO_TO_PBS_PKL}")
    print(f"[config] TARGET_RIDGE={TARGET_RIDGE:.3e}")
    print(f"[config] LAMBDAS={LAMBDAS}")

    b_ridge, ridge_used = load_b_ridge(AO_TO_PBS_PKL, TARGET_RIDGE)
    print(
        f"[load] selected ridge={ridge_used:.3e} "
        f"|b_ridge|={np.linalg.norm(b_ridge):.6e} max|b_ridge|={np.max(np.abs(b_ridge)):.6e}"
    )

    t0 = time.perf_counter()
    sys = build_system()
    geo = sys["geo"]
    pbs_mol = build_pbs_mol(geo)
    coords, w, phi_geo, _phi_pbs = build_grid(geo, pbs_mol)
    phi_l1 = dft.numint.eval_ao(sys["lgeo1"], coords, deriv=0)
    phi_r1 = dft.numint.eval_ao(sys["rgeo1"], coords, deriv=0)
    s3l, s3r, s3 = compute_s3_tensors(sys, geo, pbs_mol)
    print(f"[setup] build+grid+S3 t={time.perf_counter() - t0:.1f}s")

    if b_ridge.size != s3.shape[2]:
        raise RuntimeError(
            f"b_ridge length mismatch: len={b_ridge.size}, npbs(from S3)={s3.shape[2]}"
        )

    sys["w"] = w
    sys["phi_geo"] = phi_geo
    sys["phi_l1"] = phi_l1
    sys["phi_r1"] = phi_r1
    dref = sys["daref"] + sys["dbref"]

    rows = []
    for lam in LAMBDAS:
        b = float(lam) * b_ridge
        row = evaluate_fixed_b(b, sys, s3l, s3r, s3, dref)
        row["lambda"] = float(lam)
        flag_row_stability(row)
        row["flag_any"] = False
        rows.append(row)

        print(
            f"[lambda={lam:0.2f}] L={row['L']:.8f} Ef={row['Ef']:.8f} "
            f"L1={row['L1']:.8e} Npdft={row['Npdft']:.6f} "
            f"|grad|={row['grad_norm']:.5e} t={row['scf_wall_time_s']:.1f}s"
        )

    annotate_neighbor_flags(rows)
    for row in rows:
        row["flag_any"] = any_flagged(row)

    out = {
        "fit_pickle": AO_TO_PBS_PKL,
        "target_ridge": float(TARGET_RIDGE),
        "selected_ridge": float(ridge_used),
        "lambdas": [float(x) for x in LAMBDAS],
        "rows": rows,
        "flag_thresholds": {
            "npdft_target": NPDFT_TARGET,
            "npdft_dev_warn": NPDFT_DEV_WARN,
            "grad_norm_warn": GRAD_NORM_WARN,
            "cycle_near_max_ratio": CYCLE_NEAR_MAX_RATIO,
            "abrupt_delta_ratio": ABRUPT_DELTA_RATIO,
            "abrupt_growth_ratio": ABRUPT_GROWTH_RATIO,
        },
    }

    write_csv(OUT_CSV, rows)
    atomic_pickle_save(OUT_PKL, out)
    print(f"[save] wrote {OUT_CSV}")
    print(f"[save] wrote {OUT_PKL}")
    print_summary_table(rows)
    print("=== done ===")


if __name__ == "__main__":
    main()
