# PDFT optimizer workflow (`pdftoptimizer.py`)

Partition DFT for **Al50 + O2 @ 2.4 Å** using PBS coefficients + L-BFGS-B.
Same physical setup as `2.4capdft.py`; SCF uses grid → AO potentials like the grid
PDFT loop. S3 tensors supply the optimizer gradient.

---

## 1. Physical picture

The full system is split into **left** (O-side) and **right** (Al-side) fragments.
A single partition potential \(v_p(\mathbf r)\) is chosen so fragment densities sum
to the reference density.

**Optimization variable:** PBS coefficients \(\mathbf b = (b_1,\ldots,b_{N_\mathrm{pbs}})\).

**SCF input:** AO matrices \(V^L, V^R\) (one per side), passed to `FragmentDFT` as `Vp`.

---

## 2. Objects and matrix shapes

| Symbol | Code name | Shape | Meaning |
|--------|-----------|-------|---------|
| \(\mathbf b\) | `v` | `(npbs,)` | PBS coefficients (optimizer variable) |
| \(\chi_t(\mathbf r)\) | PBS basis fn | — | Potential basis (`6-311g**` on full `geo`) |
| \(S3^L_{\mu\nu t}\) | `s3l` | `(nao_L, nao_L, npbs)` | PBS → left AO (from `compute_s3(lgeo1)`) |
| \(S3^R_{\mu\nu t}\) | `s3r` | `(nao_R, nao_R, npbs)` | PBS → right AO (`compute_s3(rgeo1)`) |
| \(S3^\mathrm{ref}_{\mu\nu t}\) | `s3` | `(nao_G, nao_G, npbs)` | PBS → full AO (`compute_s3(geo)`) |
| \(\Phi^\mathrm{pbs}_{p t}\) | `phi_pbs` | `(ngrid, npbs)` | \(\chi_t(\mathbf r_p)\) on DFT grid |
| \(\Phi^L_{p\mu}\) | `phi_l1` | `(ngrid, nao_L)` | left fragment AO on grid |
| \(\Phi^R_{p\mu}\) | `phi_r1` | `(ngrid, nao_R)` | right fragment AO on grid |
| \(\Phi^G_{p\mu}\) | `phi_geo` | `(ngrid, nao_G)` | full system AO on grid |
| \(w_p\) | `w` | `(ngrid,)` | grid quadrature weights |
| \(v_p\) | `v_grid` | `(ngrid,)` | \(v_p(\mathbf r_p)\) |
| \(V^L_{\mu\nu}\) | `Vpl` / `vl` | `(nao_L, nao_L)` | partition potential, left AO |
| \(V^R_{\mu\nu}\) | `Vpr` / `vr` | `(nao_R, nao_R)` | partition potential, right AO |
| \(V^\mathrm{ref}_{\mu\nu}\) | `Vpref` | `(nao_G, nao_G)` | partition potential, full AO |
| \(D^L_\alpha, D^L_\beta\) | `Dal`, `Dbl` | `(nao_L, nao_L)` | left ensemble spin DMs |
| \(D^R_\alpha, D^R_\beta\) | `Dar`, `Dbr` | `(nao_R, nao_R)` | right ensemble spin DMs |
| \(D^\mathrm{ref}_\alpha, D^\mathrm{ref}_\beta\) | `daref`, `dbref` | `(nao_G, nao_G)` | reference spin DMs (from dftref pkl) |
| \(D^\mathrm{ref}\) | `dref` | `(nao_G, nao_G)` | `daref + dbref` |

Subscripts: \(\mu,\nu\) = AO indices; \(t\) = PBS index; \(p\) = grid point.

---

## 3. End-to-end workflow

```
┌─────────────────────────────────────────────────────────────────┐
│ SETUP (once)                                                    │
│  • Build fragments l, r (same as 2.4capdft.py)                  │
│  • Load D^ref from ../2.4dftref/al2.4_sigma0.002_last_dm.pkl    │
│  • Build DFT grid (level 3) → w, phi_pbs, phi_l1, phi_r1, phi_geo│
│  • compute_s3 → s3l, s3r, s3 (density-fitted, ref=2)            │
│  • Warm-start: load pkl → dm_ig1/2, project Vpl/Vpr → b, AO SCF │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ EACH L-BFGS-B STEP (Lagrangian)                                 │
│  b → v_grid → V^L, V^R, V^ref  (grid path, for SCF)             │
│  SCF → D^L, D^R, E_f                                            │
│  L(b), grad L(b)  (S3 for gradient)                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ END: compare V^AO from S3 vs grid for final b                   │
└─────────────────────────────────────────────────────────────────┘
```

**Submit:** `sbatch job_pdftoptimizer.sh`

---

## 4. Math: Vp on fragments

### Warm-start (once, capdft-compatible)

From pickle grid `vp[p]` → `v2V` → `Vpl`, `Vpr` → fragment SCF.  
Or stored `Vpl`/`Vpr` AO matrices from the pickle.

### Each optimizer step (S3 → AO)

\[
V_{\mu\nu} = \sum_t b_t\, S3_{\mu\nu t}
\]

**Code:** `vp_ao_from_pbs_coeffs(b, s3l/s3r/s3)` — same representation as the gradient.

### Grid `v2V` (diagnostics only)

Still built for warm-start and the end-of-run S3 vs grid comparison; **not** used in the L-BFGS-B loop.

### 4.1 Fragment SCF and \(E_f\)

`FragmentDFT` adds \(V\) to the core Hamiltonian in AO space:

\[
h \leftarrow h_\mathrm{core} + V
\]

Fragment energy **without** double-counting \(v_p\):

\[
E_f = E' - \sum_{\mu\nu} (D_\alpha + D_\beta)_{\mu\nu}\, V_{\mu\nu}
\]

(`get_E()` after `kernel(Vp=V)`).

Total fragment energy:

\[
E_f = E_L + E_R
\]

**Code:** `_pdft_scf_ef(l, r, vl, vr)` → `l.scf(vl)`, `r.scf(vr)`, sum `get_E()`.

---

## 5. Math: PBS → AO via S3 (gradient + diagnostics)

### 5.1 Direct AO construction

\[
V_{\mu\nu} = \sum_t b_t\, S3_{\mu\nu t}
\]

**Code:** `vp_ao_from_pbs_coeffs(b, s3)` → symmetrized.

This is **not** used for SCF in the current workflow; it is used for:

- warm-start projection (`Vpl`/`Vpr` → \(\mathbf b\))
- end-of-run comparison (S3 vs grid)

S3 is built by `n2v_pyscf_engine_df_option.compute_s3` (Coulomb-metric density fitting).

### 5.2 S3 vs grid (why they differ)

In the exact limit (exact integrals, complete bases, same \(\chi_t\)):

\[
\sum_t b_t S3_{\mu\nu t} \;\approx\; \frac{1}{N_\mathrm{grid}}\sum_p w_p \Big(\sum_t b_t\chi_t(\mathbf r_p)\Big) \phi_{p\mu}\phi_{p\nu}
\]

In practice: DF error in S3, finite grid, PBS evaluated on `geo` vs S3 built on fragment mols.

---

## 6. Lagrangian and gradient (optimizer)

### 6.1 Lagrangian

After SCF with \(V^L, V^R\) from the **grid path**:

\[
\mathcal L = E_f
+ \mathrm{Tr}\big[(D^L_\alpha + D^L_\beta) V^L\big]
+ \mathrm{Tr}\big[(D^R_\alpha + D^R_\beta) V^R\big]
- \mathrm{Tr}\big[D^\mathrm{ref}\, V^\mathrm{ref}\big]
\]

**Component form** (e.g. left trace):

\[
\mathrm{Tr}(D V) = \sum_{\mu\nu} D_{\mu\nu}\, V_{\mu\nu}
\]

L-BFGS-B **minimizes** \(-\mathcal L\) with analytic Jacobian \(-\nabla_{\mathbf b}\mathcal L\).

### 6.2 Gradient w.r.t. PBS coefficients

\[
\frac{\partial \mathcal L}{\partial b_t}
= \sum_{\mu\nu} (D^L_\alpha + D^L_\beta)_{\mu\nu}\, S3^L_{\mu\nu t}
+ \sum_{\mu\nu} (D^R_\alpha + D^R_\beta)_{\mu\nu}\, S3^R_{\mu\nu t}
- \sum_{\mu\nu} D^\mathrm{ref}_{\mu\nu}\, S3^\mathrm{ref}_{\mu\nu t}
\]

**Code:**

```python
grad = (
    einsum("ij,ijt->t", Dal + Dbl, s3l)
  + einsum("ij,ijt->t", Dar + Dbr, s3r)
  - einsum("ij,ijt->t", dref, s3)
)
```

Note: \(\mathcal L\) uses \(V\) from **grid**; \(\partial\mathcal L/\partial b_t\) uses **S3**.
This matches the h2pluspdft / n2v PBS optimizer pattern.

---

## 7. Warm-start

| Priority | File | What is used |
|----------|------|--------------|
| 1 | `pdft_optimizer.pkl` | previous optimizer run |
| 2 | `pdft_checkpointnewref6.pkl` | grid PDFT from `2.4capdft.py` |

**Fragment DMs:** `dm_ig1` → `rdft1.D`, `dm_ig2` → `rdft2.D`

**Initial \(\mathbf b\):**

1. If `vp` in pkl has length `npbs` → use as PBS coeffs
2. Else project AO matrices: least squares \(V \approx \sum_t b_t S3_{\cdot\cdot t}\)
   - from `Vpl` with `s3l`, `Vpr` with `s3r`, then average
3. Else \(\mathbf b = \mathbf 0\)

**AO warm-start SCF:** `l.scf(Vpl_pkl)`, `r.scf(Vpr_pkl)` directly (exact match to grid run).

---

## 8. Relation to `2.4capdft.py`

| | `2.4capdft.py` | `pdftoptimizer.py` |
|---|----------------|---------------------|
| \(v_p\) storage | grid vector `vp[p]` | PBS coeffs `b` |
| Update | steepest descent on grid | L-BFGS-B on `b` |
| \(V^L, V^R\) | `v2V(vp, phi_l1/r1)` | `v2V(phi_pbs @ b, phi_l1/r1)` |
| Density constraint | explicit \(\|n_f - n_\mathrm{ref}\|_1\) | encoded in \(\mathcal L\), grad |
| S3 | not used | gradient (+ warm-start projection) |

When `2.4capdft` stores `vp[p]` on the grid, the optimizer warm-start uses `Vpl`/`Vpr`
(AO) from the same run, not the raw grid vector directly (unless lengths match by chance).

---

## 9. End-of-run comparison

After `minimize` finishes, for final \(\mathbf b^\*\):

\[
V^\mathrm{S3} = \sum_t b^*_t S3_{\cdot\cdot t}, \qquad
V^\mathrm{grid} = \mathrm{v2V}(\Phi^\mathrm{pbs}\mathbf b^*, \phi)
\]

Printed for left, right, and full:

- \(\|V^\mathrm{S3} - V^\mathrm{grid}\|_F\)
- relative error vs \(\|V^\mathrm{grid}\|_F\)
- \(\max |V^\mathrm{S3} - V^\mathrm{grid}|\)

Small values → S3 and grid paths agree for the converged \(\mathbf b^\*\).

---

## 10. File map

| File | Role |
|------|------|
| `pdftoptimizer.py` | main script |
| `job_pdftoptimizer.sh` | Slurm job |
| `n2v_pyscf_engine_df_option.py` | S3 via density fitting |
| `fragments.py` | `FragmentDFT`, `ens`, `Vp` in AO |
| `pdft_optimizer.pkl` | optimizer checkpoint |
| `pdft_checkpointnewref6.pkl` | grid PDFT warm-start |
| `../2.4dftref/al2.4_sigma0.002_last_dm.pkl` | reference \(D^\mathrm{ref}\) |
