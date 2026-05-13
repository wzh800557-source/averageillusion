#!/usr/bin/env python3
"""
Paper 7: Pantheon+ Full Reanalysis — Undo Mass Step
=====================================================
1. Read Pantheon+SH0ES distance moduli (WITH mass step applied)
2. UNDO the Brout+2022 mass step (γ = 0.054 mag)
3. Read the full STAT+SYS covariance matrix
4. Fit (Ωm, M_offset) in flat ΛCDM to each mass subsample
5. Compare: Δ(Ωm) between high-mass and low-mass hosts

If Δ(Ωm) ≠ 0 after undoing the step, that's the z-dependent kernel artifact.

Data files needed (in ~/pantheonplus/):
  Pantheon+SH0ES.dat
  Pantheon+SH0ES_STAT+SYS.cov

Download:
  cd ~/pantheonplus
  curl -L 'https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat' -o Pantheon+SH0ES.dat
  curl -L 'https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov' -o 'Pantheon+SH0ES_STAT+SYS.cov'
"""

import numpy as np
import os, json, sys
from scipy.optimize import minimize
from scipy.integrate import quad

DATADIR = os.path.expanduser('~/pantheonplus')
OUTDIR = os.path.expanduser('~/paper7_results')
os.makedirs(OUTDIR, exist_ok=True)

# ── Read SNe data ──
dat_file = os.path.join(DATADIR, 'Pantheon+SH0ES.dat')
cov_file = os.path.join(DATADIR, 'Pantheon+SH0ES_STAT+SYS.cov')

if not os.path.exists(dat_file):
    print(f"ERROR: {dat_file} not found. Download it first (see docstring).")
    sys.exit(1)

print("Reading Pantheon+SH0ES data...", flush=True)
with open(dat_file, 'r') as f:
    header = f.readline().strip().split()
    rows = []
    for line in f:
        vals = line.strip().split()
        if len(vals) == len(header):
            rows.append(vals)

data = {}
for i, col in enumerate(header):
    vals = [row[i] for row in rows]
    try:
        data[col] = np.array(vals, dtype=float)
    except:
        data[col] = np.array(vals)

N_all = len(data['zHD'])
print(f"  Loaded {N_all} SNe")

# ── Read covariance matrix ──
print("Reading covariance matrix...", flush=True)
if os.path.exists(cov_file):
    with open(cov_file, 'r') as f:
        first_line = f.readline().strip()
        N_cov = int(first_line)
        print(f"  Covariance matrix size: {N_cov}×{N_cov}")
        cov_flat = []
        for line in f:
            cov_flat.extend([float(x) for x in line.strip().split()])
    
    if len(cov_flat) == N_cov * N_cov:
        C_full = np.array(cov_flat).reshape(N_cov, N_cov)
        has_cov = True
        print(f"  Covariance loaded: {C_full.shape}")
    else:
        print(f"  WARNING: Expected {N_cov**2} entries, got {len(cov_flat)}. Using diagonal.")
        has_cov = False
else:
    print(f"  WARNING: Covariance file not found. Using diagonal errors.")
    has_cov = False

# ── Quality cuts ──
z = data['zHD']
mu_sh0es = data['MU_SH0ES']
mu_err = data.get('MU_SH0ES_ERR_DIAG', np.full(N_all, 0.15))
logmass = data['HOST_LOGMASS']

good = (logmass > 5) & (logmass < 14) & (z > 0.01) & (z < 2.5) & np.isfinite(mu_sh0es)
idx_good = np.where(good)[0]
N = int(np.sum(good))

z_g = z[good]
mu_g = mu_sh0es[good]
err_g = mu_err[good]
lm_g = logmass[good]

print(f"  After cuts: {N} SNe")

# Extract sub-covariance for good SNe
if has_cov and N_cov == N_all:
    C = C_full[np.ix_(idx_good, idx_good)]
    print(f"  Sub-covariance: {C.shape}")
elif has_cov:
    print(f"  WARNING: Cov size {N_cov} != data size {N_all}. Using diagonal.")
    C = np.diag(err_g**2)
else:
    C = np.diag(err_g**2)

# ── Undo the mass step ──
# Brout+2022 (arXiv:2202.04077) Table 2: γ = 0.054 mag
# Convention: δ_host = γ/2 for logM > 10, -γ/2 for logM < 10
# MU_SH0ES has this SUBTRACTED, so to undo:
GAMMA = 0.054
MASS_CUT = 10.0

step_correction = np.where(lm_g >= MASS_CUT, -GAMMA/2, +GAMMA/2)
mu_no_step = mu_g + step_correction  # undo the mass step

print(f"\n  Undid mass step: γ = {GAMMA} mag at logM* = {MASS_CUT}")
print(f"  High-mass shifted by +{GAMMA/2:.3f} mag, low-mass by -{GAMMA/2:.3f} mag")

# ── ΛCDM distance modulus ──
def mu_lcdm(z_arr, Om):
    """Flat ΛCDM distance modulus (up to additive M)."""
    mu_arr = np.zeros_like(z_arr)
    for i, zi in enumerate(z_arr):
        if zi < 1e-6:
            mu_arr[i] = -99
            continue
        dL, _ = quad(lambda zz: 1.0/np.sqrt(Om*(1+zz)**3 + (1-Om)), 0, zi, limit=100)
        mu_arr[i] = 5 * np.log10((1+zi) * dL)
    return mu_arr

# ── Fit function ──
def fit_Om(z_sub, mu_sub, C_sub, label=""):
    """Fit (Om, M_offset) using full covariance. Returns Om, M, χ²."""
    # Invert covariance
    try:
        C_inv = np.linalg.inv(C_sub)
    except np.linalg.LinAlgError:
        # Regularise
        C_reg = C_sub + 1e-6 * np.eye(len(C_sub))
        C_inv = np.linalg.inv(C_reg)
    
    def chi2(params):
        Om, M = params
        if Om < 0.01 or Om > 0.99:
            return 1e10
        mu_th = mu_lcdm(z_sub, Om) + M
        delta = mu_sub - mu_th
        return float(delta @ C_inv @ delta)
    
    # Grid search for starting point
    best = None
    for Om0 in [0.2, 0.25, 0.3, 0.35, 0.4]:
        mu_th = mu_lcdm(z_sub, Om0)
        M0 = np.median(mu_sub - mu_th)
        res = minimize(chi2, [Om0, M0], method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-6})
        if best is None or res.fun < best.fun:
            best = res
    
    Om_fit, M_fit = best.x
    chi2_min = best.fun
    ndof = len(z_sub) - 2
    
    # Error estimate from Hessian (approximate)
    from scipy.optimize import approx_fprime
    h = 1e-4
    H = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            def f_ij(p):
                pp = best.x.copy()
                pp[i] = p
                return chi2(pp)
            if i == j:
                fp = chi2(best.x + h*np.eye(2)[i])
                fm = chi2(best.x - h*np.eye(2)[i])
                f0 = chi2_min
                H[i,j] = (fp + fm - 2*f0) / h**2
    
    sigma_Om = 1.0/np.sqrt(max(H[0,0], 1e-10)) if H[0,0] > 0 else 0.1
    
    print(f"  {label:20s}: Ωm = {Om_fit:.4f} ± {sigma_Om:.4f}, "
          f"χ²/dof = {chi2_min:.1f}/{ndof} = {chi2_min/ndof:.3f}")
    
    return Om_fit, sigma_Om, M_fit, chi2_min, ndof

# ── Split and fit ──
low = lm_g < MASS_CUT
high = lm_g >= MASS_CUT

print(f"\n  Low-mass:  {np.sum(low)} SNe, <logM*> = {lm_g[low].mean():.2f}")
print(f"  High-mass: {np.sum(high)} SNe, <logM*> = {lm_g[high].mean():.2f}")

# === Analysis A: WITH mass step (as published) ===
print(f"\n{'='*60}")
print("A. WITH mass step (MU_SH0ES as published)")
print(f"{'='*60}")

idx_low = np.where(low)[0]
idx_high = np.where(high)[0]

print("\nFitting (this takes ~5 min per subsample)...", flush=True)
Om_full_A, sOm_full_A, _, _, _ = fit_Om(z_g, mu_g, C, "Full sample")
Om_low_A, sOm_low_A, _, _, _ = fit_Om(z_g[low], mu_g[low], C[np.ix_(idx_low,idx_low)], "Low-mass hosts")
Om_high_A, sOm_high_A, _, _, _ = fit_Om(z_g[high], mu_g[high], C[np.ix_(idx_high,idx_high)], "High-mass hosts")

dOm_A = Om_high_A - Om_low_A
dOm_err_A = np.sqrt(sOm_high_A**2 + sOm_low_A**2)
print(f"\n  ΔΩm (high - low) = {dOm_A:+.4f} ± {dOm_err_A:.4f}  ({abs(dOm_A)/dOm_err_A:.1f}σ)")

# === Analysis B: WITHOUT mass step (undone) ===
print(f"\n{'='*60}")
print("B. WITHOUT mass step (undone)")
print(f"{'='*60}")

print("\nFitting...", flush=True)
Om_full_B, sOm_full_B, _, _, _ = fit_Om(z_g, mu_no_step, C, "Full (no step)")
Om_low_B, sOm_low_B, _, _, _ = fit_Om(z_g[low], mu_no_step[low], C[np.ix_(idx_low,idx_low)], "Low-mass (no step)")
Om_high_B, sOm_high_B, _, _, _ = fit_Om(z_g[high], mu_no_step[high], C[np.ix_(idx_high,idx_high)], "High-mass (no step)")

dOm_B = Om_high_B - Om_low_B
dOm_err_B = np.sqrt(sOm_high_B**2 + sOm_low_B**2)
print(f"\n  ΔΩm (high - low) = {dOm_B:+.4f} ± {dOm_err_B:.4f}  ({abs(dOm_B)/dOm_err_B:.1f}σ)")

# === Summary ===
print(f"\n{'='*60}")
print("SUMMARY: Averaging Illusion in Pantheon+")
print(f"{'='*60}")
print(f"  With mass step:    ΔΩm = {dOm_A:+.4f} ± {dOm_err_A:.4f}")
print(f"  Without mass step: ΔΩm = {dOm_B:+.4f} ± {dOm_err_B:.4f}")
print(f"  Difference:        ΔΔΩm = {dOm_B - dOm_A:+.4f}")
print()
print(f"  Kernel prediction: constant mass step cannot capture")
print(f"  z-dependent host mass evolution → residual ΔΩm ≠ 0")
print(f"  after step removal signals the averaging illusion.")

# ── Redshift-binned residuals (for a figure) ──
print(f"\n{'='*60}")
print("Redshift-binned Hubble residuals (without mass step)")
print(f"{'='*60}")

# Compute residuals relative to best-fit ΛCDM
mu_ref = mu_lcdm(z_g, Om_full_B)
M_ref = np.median(mu_no_step - mu_ref)
resid = mu_no_step - mu_ref - M_ref

z_edges = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.3]
print(f"{'z_bin':>12} {'N_lo':>5} {'N_hi':>5} {'<Δμ>_lo':>9} {'<Δμ>_hi':>9} {'Δ':>9} {'σ':>7}")

bin_data = []
for k in range(len(z_edges)-1):
    zlo, zhi = z_edges[k], z_edges[k+1]
    in_bin = (z_g >= zlo) & (z_g < zhi)
    in_lo = in_bin & low
    in_hi = in_bin & high
    nl, nh = np.sum(in_lo), np.sum(in_hi)
    if nl < 3 or nh < 3:
        continue
    
    wl = 1/err_g[in_lo]**2
    wh = 1/err_g[in_hi]**2
    ml = np.average(resid[in_lo], weights=wl)
    mh = np.average(resid[in_hi], weights=wh)
    el = 1/np.sqrt(np.sum(wl))
    eh = 1/np.sqrt(np.sum(wh))
    d = mh - ml
    de = np.sqrt(el**2 + eh**2)
    
    print(f"  {zlo:.2f}-{zhi:.2f} {nl:5d} {nh:5d} {ml:+9.4f} {mh:+9.4f} {d:+9.4f} {de:7.4f}")
    bin_data.append({'z_lo': zlo, 'z_hi': zhi, 'z_mid': (zlo+zhi)/2,
                     'n_low': int(nl), 'n_high': int(nh),
                     'mu_res_low': float(ml), 'mu_res_high': float(mh),
                     'delta': float(d), 'delta_err': float(de)})

# ── Save ──
results = {
    'N_total': N,
    'N_low': int(np.sum(low)),
    'N_high': int(np.sum(high)),
    'gamma_undone': GAMMA,
    'with_step': {
        'Om_full': float(Om_full_A), 'Om_full_err': float(sOm_full_A),
        'Om_low': float(Om_low_A), 'Om_low_err': float(sOm_low_A),
        'Om_high': float(Om_high_A), 'Om_high_err': float(sOm_high_A),
        'delta_Om': float(dOm_A), 'delta_Om_err': float(dOm_err_A),
    },
    'without_step': {
        'Om_full': float(Om_full_B), 'Om_full_err': float(sOm_full_B),
        'Om_low': float(Om_low_B), 'Om_low_err': float(sOm_low_B),
        'Om_high': float(Om_high_B), 'Om_high_err': float(sOm_high_B),
        'delta_Om': float(dOm_B), 'delta_Om_err': float(dOm_err_B),
    },
    'binned_residuals': bin_data,
    'uses_full_covariance': has_cov,
}

outpath = os.path.join(OUTDIR, 'pantheon_full_analysis.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {outpath}")
