"""
Single fixed-vp diagnostic from ao_to_pbs_fit.pkl.

Workflow:
1) Load b from ao_to_pbs_fit.pkl.
2) Reconstruct AO Vp matrices via S3 contractions.
3) Run one fragment SCF with fixed Vp.
4) Report L1/L1_alpha/L1_beta, Ef, L, and electron numbers.
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


AO_TO_PBS_PKL = "ao_to_pbs_fit.pkl"
OUT_PKL = "ao_to_pbs_fixed_vp_compare.pkl"


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


def main():
    print("=== Fixed-vp diagnostic from ao_to_pbs_fit.pkl ===")
    print(f"[config] AO_TO_PBS_PKL={AO_TO_PBS_PKL}")
    print("[config] refDM source: ../2.7dftref/al2.7_sigma0.002_last_dm.pkl (via build_system)")

    sys = build_system()
    geo = sys["geo"]
    pbs_mol = build_pbs_mol(geo)
    coords, w, phi_geo, _phi_pbs = build_grid(geo, pbs_mol)
    phi_l1 = dft.numint.eval_ao(sys["lgeo1"], coords, deriv=0)
    phi_r1 = dft.numint.eval_ao(sys["rgeo1"], coords, deriv=0)
    s3l, s3r, s3 = compute_s3_tensors(sys, geo, pbs_mol)

    with open(AO_TO_PBS_PKL, "rb") as f:
        data_fit = pickle.load(f)
    if "vp" not in data_fit:
        raise KeyError(f"{AO_TO_PBS_PKL} does not contain key 'vp'")

    b = np.asarray(data_fit["vp"], dtype=np.float64).ravel()
    Vpl = vp_ao_from_pbs_coeffs(b, s3l)
    Vpr = vp_ao_from_pbs_coeffs(b, s3r)
    Vpref = vp_ao_from_pbs_coeffs(b, s3)

    print(
        f"[load] b len={b.size} |b|={np.linalg.norm(b):.6e} "
        f"max|b|={np.max(np.abs(b)):.6e}"
    )

    metrics = fixed_vp_scf_metrics(
        build_system(),
        Vpl,
        Vpr,
        Vpref,
        s3l,
        s3r,
        s3,
        phi_l1,
        phi_r1,
        phi_geo,
        w,
    )
    print(
        f"[reconstructed AO] L1={metrics['L1']:.8e} L1_alpha={metrics['L1_alpha']:.8e} "
        f"L1_beta={metrics['L1_beta']:.8e} Ef={metrics['Ef']:.8f} L={metrics['L']:.8f} "
        f"N=({metrics['N_alpha']:.8f},{metrics['N_beta']:.8f},{metrics['N_total']:.8f}) "
        f"|grad|={metrics['grad_norm']:.6e}"
    )

    out = {
        "fit_pickle": AO_TO_PBS_PKL,
        "b_norm": float(np.linalg.norm(b)),
        "b_max_abs": float(np.max(np.abs(b))),
        "fixed_vp_scf_reconstructed": metrics,
    }
    atomic_pickle_save(OUT_PKL, out)
    print(f"[save] wrote {OUT_PKL}")
    print("=== done ===")


if __name__ == "__main__":
    main()
