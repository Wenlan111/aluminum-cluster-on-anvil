"""
Standalone AO -> PBS diagnostic.

This script compares an old converged AO potential checkpoint against the
current PBS/S3 representation, without running SCF or any optimizer.
"""

import pickle

import numpy as np
from pyscf import dft

from pdftoptimizer import (
    atomic_pickle_save,
    build_grid,
    build_pbs_mol,
    build_system,
    compute_s3_tensors,
    v2V,
    vp_ao_from_pbs_coeffs,
    vp_grid_from_pbs,
)


OLD_PKL = "pdft2.7_checkpointnewref.pkl"
AO_TO_PBS_OUT = "ao_to_pbs_fit.pkl"
RIDGE = 1e-8
RIDGE_LIST = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def flatten_sym_matrix(M):
    M = np.asarray(M, dtype=np.float64)
    M = 0.5 * (M + M.T)
    idx = np.triu_indices(M.shape[0])
    return M[idx], idx


def flatten_sym_s3(S3, idx):
    # return shape (n_upper, npbs)
    S3 = np.asarray(S3, dtype=np.float64)
    return S3[idx[0], idx[1], :]


def fit_b_from_ao_cases(cases, ridge=1e-8):
    # cases is list of tuples:
    # (name, V_old, S3, weight)
    #
    # Use only upper triangle to avoid double-counting symmetric AO matrices.
    # Build stacked least squares:
    # A_all b ≈ y_all
    #
    # If ridge > 0:
    #   solve (A.T @ A + ridge I) b = A.T @ y
    # Else:
    #   use np.linalg.lstsq(A, y, rcond=1e-10)
    #
    # Print A shape, condition info if available, |b|, max|b|.
    # Return b.
    if not cases:
        raise ValueError("fit_b_from_ao_cases: no AO cases were provided")

    A_parts = []
    y_parts = []
    npbs_ref = None

    print("\n[fit] building stacked AO->PBS least-squares system")
    for name, V_old, S3, weight in cases:
        y_case, idx = flatten_sym_matrix(V_old)
        A_case = flatten_sym_s3(S3, idx)
        if npbs_ref is None:
            npbs_ref = A_case.shape[1]
        elif A_case.shape[1] != npbs_ref:
            raise ValueError(
                f"case '{name}' has npbs={A_case.shape[1]} but expected {npbs_ref}"
            )

        ws = float(np.sqrt(max(weight, 0.0)))
        A_parts.append(ws * A_case)
        y_parts.append(ws * y_case)
        print(
            f"[fit] case={name:<5} A={A_case.shape} "
            f"||V_old||={np.linalg.norm(V_old):.6e} weight={weight:.3g}"
        )

    A_all = np.vstack(A_parts)
    y_all = np.concatenate(y_parts)
    print(f"[fit] stacked A shape={A_all.shape}, y shape={y_all.shape}")

    AtA = A_all.T @ A_all
    Aty = A_all.T @ y_all
    cond_ata = np.linalg.cond(AtA)
    print(f"[fit] cond(A^T A) = {cond_ata:.6e}")

    if ridge > 0:
        lhs = AtA + float(ridge) * np.eye(AtA.shape[0], dtype=np.float64)
        b = np.linalg.solve(lhs, Aty)
        print(f"[fit] solved normal equations with ridge={ridge:.3e}")
    else:
        b, residuals, rank, svals = np.linalg.lstsq(A_all, y_all, rcond=1e-10)
        print(
            f"[fit] solved via lstsq, rank={rank}, "
            f"min/max singular={svals.min():.6e}/{svals.max():.6e}, "
            f"residual_vec_len={len(residuals)}"
        )

    b = np.asarray(b, dtype=np.float64).ravel()
    print(f"[fit] |b|={np.linalg.norm(b):.6e}, max|b|={np.max(np.abs(b)):.6e}")
    return b


def ao_fit_diagnostics(cases, b):
    summary = {}
    print("\n=== AO fit diagnostics ===")
    for name, V_old, S3, _weight in cases:
        V_old = np.asarray(V_old, dtype=np.float64)
        V_fit = vp_ao_from_pbs_coeffs(b, S3)
        diff = V_fit - V_old
        n_old = np.linalg.norm(V_old)
        n_fit = np.linalg.norm(V_fit)
        abs_err = np.linalg.norm(diff)
        rel_err = abs_err / n_old if n_old > 0 else np.nan
        maxabs = np.max(np.abs(diff))
        print(
            f"[AO fit] {name:<5} rel={rel_err:.6e} abs={abs_err:.6e} "
            f"||V_old||={n_old:.6e} ||V_fit||={n_fit:.6e} max|diff|={maxabs:.6e}"
        )
        summary[name] = {
            "rel_error": float(rel_err),
            "abs_error": float(abs_err),
            "norm_old": float(n_old),
            "norm_fit": float(n_fit),
            "max_abs_diff": float(maxabs),
        }
    return summary


def s3_vs_grid_diagnostics(b, w, phi_pbs, phi_l1, phi_r1, phi_geo, s3l, s3r, s3):
    summary = {}
    v_grid = vp_grid_from_pbs(b, phi_pbs)
    cases = (
        ("left", s3l, phi_l1),
        ("right", s3r, phi_r1),
        ("full", s3, phi_geo),
    )

    print("\n=== S3 vs grid diagnostics ===")
    for name, s3_frag, phi_frag in cases:
        V_s3 = vp_ao_from_pbs_coeffs(b, s3_frag)
        V_grid = v2V(v_grid, w, phi_frag)
        diff = V_s3 - V_grid
        n_grid = np.linalg.norm(V_grid)
        n_s3 = np.linalg.norm(V_s3)
        rel = np.linalg.norm(diff) / n_grid if n_grid > 0 else np.nan
        maxabs = np.max(np.abs(diff))
        print(
            f"[S3 vs grid] {name:<5} rel={rel:.6e} ||S3||={n_s3:.6e} "
            f"||grid||={n_grid:.6e} max|diff|={maxabs:.6e}"
        )
        summary[name] = {
            "rel_error": float(rel),
            "norm_s3": float(n_s3),
            "norm_grid": float(n_grid),
            "max_abs_diff": float(maxabs),
        }
    return summary


def summarize_ao_tradeoff(ao_summary, b):
    rel_errors = np.array(
        [entry["rel_error"] for entry in ao_summary.values()], dtype=np.float64
    )
    return {
        "rel_mean": float(np.mean(rel_errors)),
        "rel_max": float(np.max(rel_errors)),
        "b_norm": float(np.linalg.norm(b)),
        "b_max_abs": float(np.max(np.abs(b))),
    }


def main():
    print("=== AO -> PBS diagnostic (no SCF / no optimizer) ===")
    print(f"[config] OLD_PKL={OLD_PKL}")
    print(f"[config] AO_TO_PBS_OUT={AO_TO_PBS_OUT}")
    print(f"[config] RIDGE (selected output)={RIDGE:.3e}")
    print(f"[config] RIDGE_LIST={RIDGE_LIST}")

    sys = build_system()
    geo = sys["geo"]
    pbs_mol = build_pbs_mol(geo)
    coords, w, phi_geo, phi_pbs = build_grid(geo, pbs_mol)
    phi_l1 = dft.numint.eval_ao(sys["lgeo1"], coords, deriv=0)
    phi_r1 = dft.numint.eval_ao(sys["rgeo1"], coords, deriv=0)
    s3l, s3r, s3 = compute_s3_tensors(sys, geo, pbs_mol)

    print(
        f"[setup] grid: ngrid={coords.shape[0]}, nao_full={phi_geo.shape[1]}, "
        f"npbs={phi_pbs.shape[1]}"
    )
    print(
        f"[setup] S3 shapes left={s3l.shape}, right={s3r.shape}, full={s3.shape}"
    )

    with open(OLD_PKL, "rb") as f:
        data = pickle.load(f)
    print(f"[load] loaded keys: {sorted(list(data.keys()))}")

    cases = []
    if "Vpl" in data:
        cases.append(("left", data["Vpl"], s3l, 1.0))
    if "Vpr" in data:
        cases.append(("right", data["Vpr"], s3r, 1.0))
    if "Vpref" in data:
        cases.append(("full", data["Vpref"], s3, 1.0))

    if not cases:
        print(
            "[error] no AO matrices found in checkpoint. Need at least one of: "
            "'Vpl', 'Vpr', 'Vpref'."
        )
        return

    print("\n=== Ridge scan: AO fit vs coefficient norm tradeoff ===")
    scan_results = []
    selected = None
    for ridge in RIDGE_LIST:
        print(f"\n--- ridge = {ridge:.3e} ---")
        b_try = fit_b_from_ao_cases(cases, ridge=ridge)
        ao_summary_try = ao_fit_diagnostics(cases, b_try)
        tradeoff = summarize_ao_tradeoff(ao_summary_try, b_try)
        print(
            f"[tradeoff] ridge={ridge:.3e} "
            f"AO rel(mean/max)=({tradeoff['rel_mean']:.6e}/{tradeoff['rel_max']:.6e}) "
            f"|b|={tradeoff['b_norm']:.6e} max|b|={tradeoff['b_max_abs']:.6e}"
        )
        scan_results.append(
            {
                "ridge": float(ridge),
                "b_norm": tradeoff["b_norm"],
                "b_max_abs": tradeoff["b_max_abs"],
                "ao_rel_mean": tradeoff["rel_mean"],
                "ao_rel_max": tradeoff["rel_max"],
                "ao_fit": ao_summary_try,
                "vp": b_try,
            }
        )
        if np.isclose(ridge, RIDGE):
            selected = {
                "ridge": float(ridge),
                "vp": b_try,
                "ao_fit": ao_summary_try,
                "tradeoff": tradeoff,
            }

    if selected is None:
        # Fallback if RIDGE is not in RIDGE_LIST.
        print(
            f"[warn] selected RIDGE={RIDGE:.3e} not in RIDGE_LIST; "
            "computing selected fit separately."
        )
        b = fit_b_from_ao_cases(cases, ridge=RIDGE)
        ao_summary = ao_fit_diagnostics(cases, b)
        selected = {
            "ridge": float(RIDGE),
            "vp": b,
            "ao_fit": ao_summary,
            "tradeoff": summarize_ao_tradeoff(ao_summary, b),
        }

    print("\n=== Ridge scan summary (compact) ===")
    for row in scan_results:
        print(
            f"[scan] ridge={row['ridge']:.3e} "
            f"AO rel(mean/max)=({row['ao_rel_mean']:.6e}/{row['ao_rel_max']:.6e}) "
            f"|b|={row['b_norm']:.6e} max|b|={row['b_max_abs']:.6e}"
        )

    b = selected["vp"]
    ao_summary = selected["ao_fit"]
    s3_grid_summary = s3_vs_grid_diagnostics(
        b, w, phi_pbs, phi_l1, phi_r1, phi_geo, s3l, s3r, s3
    )

    fit_summary = {
        "n_cases": len(cases),
        "cases": [name for name, _V_old, _S3, _wgt in cases],
        "ao_fit": ao_summary,
        "s3_vs_grid": s3_grid_summary,
        "b_norm": float(np.linalg.norm(b)),
        "b_max_abs": float(np.max(np.abs(b))),
        "ridge_scan": scan_results,
        "selected_ridge": float(selected["ridge"]),
        "selected_tradeoff": selected["tradeoff"],
    }

    out = {
        "vp": b,
        "source_pickle": OLD_PKL,
        "ridge": float(selected["ridge"]),
        "fit_summary": fit_summary,
    }
    atomic_pickle_save(AO_TO_PBS_OUT, out)
    print(f"[save] wrote fitted coefficients to {AO_TO_PBS_OUT}")
    print("=== done ===")


if __name__ == "__main__":
    main()
"""
Standalone AO -> PBS diagnostic (Path B only).

This script fits PBS coefficients ``b`` from stored AO targets
(``Vpl``/``Vpr``/``Vpref``) using ``S3[:,:,t]`` and scans ridge values.
No optimizer modification is involved.
"""

import pickle
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


OLD_PKL = "pdft2.7_checkpointnewref.pkl"
AO_TO_PBS_OUT = "ao_to_pbs_fit.pkl"
RIDGE = 1e-8
RIDGE_LIST = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]


def flatten_sym_matrix(M):
    M = np.asarray(M, dtype=np.float64)
    M = 0.5 * (M + M.T)
    idx = np.triu_indices(M.shape[0])
    return M[idx], idx


def flatten_sym_s3(S3, idx):
    # return shape (n_upper, npbs)
    S3 = np.asarray(S3, dtype=np.float64)
    return S3[idx[0], idx[1], :]


def fit_b_from_ao_cases(cases, ridge=1e-8):
    # cases is list of tuples:
    # (name, V_old, S3, weight)
    #
    # Use only upper triangle to avoid double-counting symmetric AO matrices.
    # Build stacked least squares:
    # A_all b ≈ y_all
    #
    # If ridge > 0:
    #   solve (A.T @ A + ridge I) b = A.T @ y
    # Else:
    #   use np.linalg.lstsq(A, y, rcond=1e-10)
    #
    # Print A shape, condition info if available, |b|, max|b|.
    # Return b.
    if not cases:
        raise ValueError("fit_b_from_ao_cases: no AO cases were provided")

    A_parts = []
    y_parts = []
    npbs_ref = None

    print("\n[fit] building stacked AO->PBS least-squares system")
    for case in cases:
        name = case["name"]
        V_old = case["V_old"]
        S3 = case["S3"]
        weight = case["weight"]
        y_case, idx = flatten_sym_matrix(V_old)
        A_case = flatten_sym_s3(S3, idx)
        if npbs_ref is None:
            npbs_ref = A_case.shape[1]
        elif A_case.shape[1] != npbs_ref:
            raise ValueError(
                f"case '{name}' has npbs={A_case.shape[1]} but expected {npbs_ref}"
            )

        ws = float(np.sqrt(max(weight, 0.0)))
        A_parts.append(ws * A_case)
        y_parts.append(ws * y_case)
        print(
            f"[fit] case={name:<5} A={A_case.shape} "
            f"||V_old||={np.linalg.norm(V_old):.6e} weight={weight:.3g}"
        )

    A_all = np.vstack(A_parts)
    y_all = np.concatenate(y_parts)
    print(f"[fit] stacked A shape={A_all.shape}, y shape={y_all.shape}")

    AtA = A_all.T @ A_all
    Aty = A_all.T @ y_all
    cond_ata = np.linalg.cond(AtA)
    print(f"[fit] cond(A^T A) = {cond_ata:.6e}")

    if ridge > 0:
        lhs = AtA + float(ridge) * np.eye(AtA.shape[0], dtype=np.float64)
        b = np.linalg.solve(lhs, Aty)
        print(f"[fit] solved normal equations with ridge={ridge:.3e}")
    else:
        b, residuals, rank, svals = np.linalg.lstsq(A_all, y_all, rcond=1e-10)
        print(
            f"[fit] solved via lstsq, rank={rank}, "
            f"min/max singular={svals.min():.6e}/{svals.max():.6e}, "
            f"residual_vec_len={len(residuals)}"
        )

    b = np.asarray(b, dtype=np.float64).ravel()
    print(f"[fit] |b|={np.linalg.norm(b):.6e}, max|b|={np.max(np.abs(b)):.6e}")
    return b


def matrix_fit_metrics(V_old, V_fit):
    V_old = np.asarray(V_old, dtype=np.float64)
    V_fit = np.asarray(V_fit, dtype=np.float64)
    diff = V_fit - V_old
    n_old = np.linalg.norm(V_old)
    n_fit = np.linalg.norm(V_fit)
    abs_err = np.linalg.norm(diff)
    rel_err = abs_err / n_old if n_old > 0 else np.nan
    maxabs = np.max(np.abs(diff))
    return {
        "rel_error": float(rel_err),
        "abs_error": float(abs_err),
        "norm_old": float(n_old),
        "norm_fit": float(n_fit),
        "max_abs_diff": float(maxabs),
    }


def block_masks_from_mol(mol):
    labels = mol.ao_labels()
    masks = {}
    for i, label in enumerate(labels):
        low = label.lower()
        if "ghost-o" in low:
            key = "ghost-O"
        elif "ghost-al" in low:
            key = "ghost-Al"
        elif " o " in low:
            key = "O2"
        elif " al " in low:
            key = "Al"
        else:
            key = "other"
        masks.setdefault(key, []).append(i)
    return {k: np.array(v, dtype=int) for k, v in masks.items() if len(v) > 0}


def block_pairs_for_masks(masks):
    pairs = []
    if "O2" in masks:
        pairs.append(("O2/O2", "O2", "O2"))
    if "Al" in masks:
        pairs.append(("Al/Al", "Al", "Al"))
    if "O2" in masks and "Al" in masks:
        pairs.append(("O2/Al", "O2", "Al"))
    if "O2" in masks and "ghost-Al" in masks:
        pairs.append(("O2/ghost-Al", "O2", "ghost-Al"))
    if "ghost-O" in masks and "Al" in masks:
        pairs.append(("ghost-O/Al", "ghost-O", "Al"))
    return pairs


def block_error_metrics(V_old, V_fit, mol):
    V_old = np.asarray(V_old, dtype=np.float64)
    V_fit = np.asarray(V_fit, dtype=np.float64)
    masks = block_masks_from_mol(mol)
    pairs = block_pairs_for_masks(masks)
    out = {}
    for label, key_i, key_j in pairs:
        ii = masks[key_i]
        jj = masks[key_j]
        old_block = V_old[np.ix_(ii, jj)]
        fit_block = V_fit[np.ix_(ii, jj)]
        diff_block = fit_block - old_block
        n_old = np.linalg.norm(old_block)
        abs_err = np.linalg.norm(diff_block)
        rel_err = abs_err / n_old if n_old > 0 else np.nan
        out[label] = {
            "rel_error": float(rel_err),
            "abs_error": float(abs_err),
            "max_abs_diff": float(np.max(np.abs(diff_block))),
            "shape": tuple(old_block.shape),
        }
    return out


def fixed_vp_scf_metrics(sys, Vpl, Vpr, Vpref, s3l, s3r, s3, phi_l1, phi_r1, phi_geo, w):
    Vpl = np.asarray(Vpl, dtype=np.float64)
    Vpr = np.asarray(Vpr, dtype=np.float64)
    Vpref = np.asarray(Vpref, dtype=np.float64)

    Ef, Dal, Dbl, Dar, Dbr = run_pdft_scf(sys["l"], sys["r"], Vpl, Vpr)
    na = density_from_dm_on_grid(phi_l1, Dal) + density_from_dm_on_grid(phi_r1, Dar)
    nb = density_from_dm_on_grid(phi_l1, Dbl) + density_from_dm_on_grid(phi_r1, Dbr)
    nref_a = density_from_dm_on_grid(phi_geo, sys["daref"])
    nref_b = density_from_dm_on_grid(phi_geo, sys["dbref"])
    nf = na + nb
    nref = nref_a + nref_b

    l1_total = float(np.sum(np.abs(nf - nref) * w))
    l1_alpha = float(np.sum(np.abs(na - nref_a) * w))
    l1_beta = float(np.sum(np.abs(nb - nref_b) * w))
    n_alpha = float(np.sum(na * w))
    n_beta = float(np.sum(nb * w))

    lag = (
        Ef
        + np.trace(Dal @ Vpl)
        + np.trace(Dbl @ Vpl)
        + np.trace(Dar @ Vpr)
        + np.trace(Dbr @ Vpr)
        - np.trace((sys["daref"] + sys["dbref"]) @ Vpref)
    )
    grad = (
        np.einsum("ij,ijt->t", Dal + Dbl, s3l)
        + np.einsum("ij,ijt->t", Dar + Dbr, s3r)
        - np.einsum("ij,ijt->t", sys["daref"] + sys["dbref"], s3)
    )
    return {
        "Ef": float(Ef),
        "L": float(lag),
        "grad_norm": float(np.linalg.norm(grad)),
        "L1": l1_total,
        "L1_alpha": l1_alpha,
        "L1_beta": l1_beta,
        "N_alpha": n_alpha,
        "N_beta": n_beta,
        "N_total": n_alpha + n_beta,
    }


def ao_fit_diagnostics(cases, b):
    summary = {}
    print("\n=== AO fit diagnostics ===")
    for case in cases:
        name = case["name"]
        V_old = np.asarray(case["V_old"], dtype=np.float64)
        S3 = case["S3"]
        V_fit = vp_ao_from_pbs_coeffs(b, S3)
        met = matrix_fit_metrics(V_old, V_fit)
        print(
            f"[AO fit] {name:<5} rel={met['rel_error']:.6e} abs={met['abs_error']:.6e} "
            f"||V_old||={met['norm_old']:.6e} ||V_fit||={met['norm_fit']:.6e} "
            f"max|diff|={met['max_abs_diff']:.6e}"
        )
        blocks = block_error_metrics(V_old, V_fit, case["mol"])
        for blk, blk_m in blocks.items():
            print(
                f"[AO block] {name:<5} {blk:<12} "
                f"rel={blk_m['rel_error']:.6e} abs={blk_m['abs_error']:.6e} "
                f"max|diff|={blk_m['max_abs_diff']:.6e} shape={blk_m['shape']}"
            )
        summary[name] = dict(met)
        summary[name]["blocks"] = blocks
    return summary


def summarize_ao_tradeoff(ao_summary, b):
    rel_errors = np.array(
        [entry["rel_error"] for entry in ao_summary.values()], dtype=np.float64
    )
    return {
        "rel_mean": float(np.mean(rel_errors)),
        "rel_max": float(np.max(rel_errors)),
        "b_norm": float(np.linalg.norm(b)),
        "b_max_abs": float(np.max(np.abs(b))),
    }


def main():
    print("=== Fixed-vp diagnostic from ao_to_pbs_fit.pkl ===")
    print(f"[config] AO_TO_PBS_OUT={AO_TO_PBS_OUT}")
    print("[config] refDM source: ../2.7dftref/al2.7_sigma0.002_last_dm.pkl (via build_system)")

    sys = build_system()
    geo = sys["geo"]
    pbs_mol = build_pbs_mol(geo)
    coords, w, phi_geo, _phi_pbs = build_grid(geo, pbs_mol)
    phi_l1 = dft.numint.eval_ao(sys["lgeo1"], coords, deriv=0)
    phi_r1 = dft.numint.eval_ao(sys["rgeo1"], coords, deriv=0)
    s3l, s3r, s3 = compute_s3_tensors(sys, geo, pbs_mol)

    with open(AO_TO_PBS_OUT, "rb") as f:
        data_fit = pickle.load(f)

    if "vp" not in data_fit:
        print("[error] AO_TO_PBS_OUT must contain vp")
        return

    b = np.asarray(data_fit["vp"], dtype=np.float64).ravel()
    Vpl_recon = vp_ao_from_pbs_coeffs(b, s3l)
    Vpr_recon = vp_ao_from_pbs_coeffs(b, s3r)
    Vpref_recon = vp_ao_from_pbs_coeffs(b, s3)

    print(
        f"[load] b from {AO_TO_PBS_OUT}: len={b.size} |b|={np.linalg.norm(b):.6e} "
        f"max|b|={np.max(np.abs(b)):.6e}"
    )

    print("\n=== Fixed-vp fragment SCF using reconstructed AO matrices from b ===")
    recon = fixed_vp_scf_metrics(
        build_system(),
        Vpl_recon,
        Vpr_recon,
        Vpref_recon,
        s3l,
        s3r,
        s3,
        phi_l1,
        phi_r1,
        phi_geo,
        w,
    )
    print(
        f"[reconstructed AO] L1={recon['L1']:.8e} L1_alpha={recon['L1_alpha']:.8e} "
        f"L1_beta={recon['L1_beta']:.8e} Ef={recon['Ef']:.8f} L={recon['L']:.8f} "
        f"N=({recon['N_alpha']:.8f},{recon['N_beta']:.8f},{recon['N_total']:.8f})"
    )

    out = {
        "fit_pickle": AO_TO_PBS_OUT,
        "b_norm": float(np.linalg.norm(b)),
        "b_max_abs": float(np.max(np.abs(b))),
        "fixed_vp_scf_reconstructed": recon,
    }
    atomic_pickle_save("ao_to_pbs_fixed_vp_compare.pkl", out)
    print("[save] wrote ao_to_pbs_fixed_vp_compare.pkl")
    print("=== done ===")


if __name__ == "__main__":
    main()
