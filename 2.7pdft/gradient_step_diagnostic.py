import os
import pickle
import time

import numpy as np

import pdftoptimizer as opt


STEP_NORMS = (0.05, 0.10, 0.20, 0.30)
DEFAULT_WARMSTART = "pdft_optimizer_trust_ridge1e34.pkl"


def load_checkpoint_b(checkpoint_path):
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    if "vp" not in data:
        raise KeyError(f"checkpoint missing 'vp': {checkpoint_path}")
    return np.asarray(data["vp"], dtype=np.float64).ravel()


def eval_state_with_l1(b, l, r, dref, s3l, s3r, s3, sys):
    b = np.asarray(b, dtype=np.float64).ravel()

    vpl = opt.vp_ao_from_pbs_coeffs(b, s3l)
    vpr = opt.vp_ao_from_pbs_coeffs(b, s3r)
    vpref = opt.vp_ao_from_pbs_coeffs(b, s3)

    t_scf = time.perf_counter()
    ef, dal, dbl, dar, dbr = opt.run_pdft_scf(l, r, vpl, vpr)
    scf_wall_s = time.perf_counter() - t_scf

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

    nl = opt.density_from_dm_on_grid(sys["phi_l1"], dal + dbl)
    nr = opt.density_from_dm_on_grid(sys["phi_r1"], dar + dbr)
    nf = nl + nr
    nref = opt.density_from_dm_on_grid(sys["phi_geo"], dref)
    l1 = float(np.sum(np.abs(nf - nref) * sys["w"]))

    return {
        "L": float(lag),
        "Ef": float(ef),
        "grad": grad,
        "L1": l1,
        "scf_wall_s": float(scf_wall_s),
    }


def main():
    opt.WARMSTART_PKL = os.environ.get("PDFT_WARMSTART_PKL", DEFAULT_WARMSTART)
    checkpoint_path = os.path.join(os.path.dirname(__file__), opt.CHECKPOINT)

    print(f"[diag] checkpoint = {checkpoint_path}")
    print(f"[diag] warm-start = {opt.WARMSTART_PKL}")
    b = load_checkpoint_b(checkpoint_path)

    t0 = time.perf_counter()
    sys = opt.build_system()
    print(f"[diag] build_system t={time.perf_counter() - t0:.1f}s")

    geo = sys["geo"]
    l = sys["l"]
    r = sys["r"]
    dref = sys["daref"] + sys["dbref"]

    print("[diag] building grid (level 3)...")
    t0 = time.perf_counter()
    pbs_mol = opt.build_pbs_mol(geo)
    coords, w, phi_geo, _ = opt.build_grid(geo, pbs_mol)
    phi_l1 = opt.dft.numint.eval_ao(sys["lgeo1"], coords, deriv=0)
    phi_r1 = opt.dft.numint.eval_ao(sys["rgeo1"], coords, deriv=0)
    sys["w"] = w
    sys["phi_geo"] = phi_geo
    sys["phi_l1"] = phi_l1
    sys["phi_r1"] = phi_r1
    print(f"[diag] grid t={time.perf_counter() - t0:.1f}s")

    print("[diag] computing S3 tensors (DF)...")
    t0 = time.perf_counter()
    s3l, s3r, s3 = opt.compute_s3_tensors(sys, geo, pbs_mol)
    print(f"[diag] S3 t={time.perf_counter() - t0:.1f}s")

    if b.size != s3.shape[2]:
        raise ValueError(f"checkpoint vp length {b.size} != npbs {s3.shape[2]}")

    warm_data, warm_path = opt.load_warmstart_pickle(sys)
    fragment_dm_guess = opt.load_fragment_dm_guess(warm_data, sys)
    if fragment_dm_guess:
        opt.apply_fragment_dm_guess(sys, fragment_dm_guess)
        print(f"[diag] applied fragment DM warm-start from {warm_path}")
    else:
        print("[diag] no fragment DM warm-start available")

    current = eval_state_with_l1(b, l, r, dref, s3l, s3r, s3, sys)
    grad_norm = float(np.linalg.norm(current["grad"]))
    if grad_norm == 0.0:
        raise ValueError("current gradient norm is zero; cannot define ghat")

    ghat = current["grad"] / grad_norm

    print(
        f"[current] L={current['L']:.8f} Ef={current['Ef']:.8f} "
        f"L1={current['L1']:.8e} |grad|={grad_norm:.5e} "
        f"SCF wall time={current['scf_wall_s']:.2f}s"
    )

    for step_norm in STEP_NORMS:
        b_trial = b + step_norm * ghat
        trial = eval_state_with_l1(b_trial, l, r, dref, s3l, s3r, s3, sys)
        dL = trial["L"] - current["L"]
        dEf = trial["Ef"] - current["Ef"]
        dL1 = trial["L1"] - current["L1"]
        grad_trial_norm = float(np.linalg.norm(trial["grad"]))
        print(
            f"step_norm={step_norm:.2f} "
            f"dL={dL:.8e} dEf={dEf:.8e} dL1={dL1:.8e} "
            f"|grad_trial|={grad_trial_norm:.5e} "
            f"SCF wall time={trial['scf_wall_s']:.2f}s"
        )


if __name__ == "__main__":
    main()
