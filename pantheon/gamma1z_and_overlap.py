#!/usr/bin/env python3
"""
Paper 7: Two Critical Referee Tests
=====================================

Test 1: Identify Pantheon+/DES-SN5YR overlap in low-z samples
      → Recompute combined significance with/without overlap

Test 2: Single-likelihood fit with shared cosmology + z-dependent mass step
      → Direct test of γ₁ ≠ 0 (the kernel evolution signal)

Run on cluster where both datasets are available.
"""

import numpy as np
import os, json, sys
from scipy.optimize import minimize
from scipy.integrate import quad

PANTHEON_DIR = os.path.expanduser('~/pantheonplus')
DES_REPO = os.path.expanduser('~/des_sn5yr/DES-SN5YR/4_DISTANCES_COVMAT')
OUTDIR = os.path.expanduser('~/paper7_results')
os.makedirs(OUTDIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# UTILITY: Read SNANA-format files
# ══════════════════════════════════════════════════════════════

def read_snana(filepath):
    data = {}
    with open(filepath, 'r') as f:
        header = None
        for line in f:
            line = line.strip()
            if line.startswith('#'): continue
            if 'VARNAMES:' in line:
                header = line.split()[1:]
                for c in header: data[c] = []
                continue
            if header and line.startswith('SN:'):
                vals = line.split()[1:]
                for c, v in zip(header, vals):
                    try: data[c].append(float(v))
                    except: data[c].append(v)
    for c in data:
        try: data[c] = np.array(data[c], dtype=float)
        except: data[c] = np.array(data[c])
    return data

def read_pantheon():
    """Read Pantheon+SH0ES data."""
    fpath = os.path.join(PANTHEON_DIR, 'Pantheon+SH0ES.dat')
    data = read_snana(fpath)
    # Read covariance
    cov_path = os.path.join(PANTHEON_DIR, 'Pantheon+SH0ES_STAT+SYS.cov')
    with open(cov_path) as f:
        N = int(f.readline().strip())
        vals = []
        for line in f:
            vals.extend([float(x) for x in line.strip().split()])
    C = np.array(vals).reshape(N, N)
    return data, C

def read_des():
    """Read DES-SN5YR data with precision matrix."""
    hd = read_snana(os.path.join(DES_REPO, 'DES-Dovekie_HD.csv'))
    meta = read_snana(os.path.join(DES_REPO, 'DES-Dovekie_Metadata.csv'))
    # Cross-match for host mass
    mass_lookup = dict(zip(meta['CID'], meta['HOST_LOGMASS']))
    hd['HOST_LOGMASS'] = np.array([mass_lookup.get(c, -9) for c in hd['CID']])
    # Load precision matrix and invert
    d = np.load(os.path.join(DES_REPO, 'STAT+SYS.npz'), allow_pickle=True)
    n = int(d[d.files[0]].item())
    P = np.zeros((n, n))
    P[np.triu_indices(n)] = d[d.files[1]]
    P[np.tril_indices(n, -1)] = P.T[np.tril_indices(n, -1)]
    C = np.linalg.inv(P)
    return hd, C

def mu_lcdm(z_arr, Om):
    out = np.zeros_like(z_arr)
    for i, zi in enumerate(z_arr):
        if zi < 1e-6: out[i] = -99; continue
        dL, _ = quad(lambda zz: 1.0/np.sqrt(Om*(1+zz)**3+(1-Om)), 0, zi, limit=100)
        out[i] = 5*np.log10((1+zi)*dL)
    return out


# ══════════════════════════════════════════════════════════════
# TEST 1: Overlap analysis
# ══════════════════════════════════════════════════════════════

def test1_overlap():
    print("="*70)
    print("TEST 1: Pantheon+ / DES-SN5YR Overlap Analysis")
    print("="*70)
    
    pp, C_pp = read_pantheon()
    des, C_des = read_des()
    
    N_pp = len(pp['zHD'])
    N_des = len(des['zHD'])
    print("  Pantheon+: %d SNe" % N_pp)
    print("  DES-SN5YR: %d SNe" % N_des)
    
    # Method 1: Match by CID/name
    pp_names = set(pp.get('CID', []))
    des_names = set(des.get('CID', []))
    name_overlap = pp_names & des_names
    print("\n  CID overlap: %d SNe" % len(name_overlap))
    if name_overlap:
        print("  Examples: %s" % sorted(list(name_overlap))[:10])
    
    # Method 2: Match by (z, survey) — low-z externals
    # Low-z SNe (z < 0.1) are likely shared between datasets
    pp_lowz = pp['zHD'] < 0.1
    des_lowz = des['zHD'] < 0.1
    print("\n  Pantheon+ low-z (z<0.1): %d" % np.sum(pp_lowz))
    print("  DES-SN5YR low-z (z<0.1): %d" % np.sum(des_lowz))
    
    # Method 3: Match by (z, mu) within tolerance
    overlap_idx_pp = []
    overlap_idx_des = []
    for i in range(N_des):
        if des['zHD'][i] > 0.1: continue  # only check low-z
        dz = np.abs(pp['zHD'] - des['zHD'][i])
        best = np.argmin(dz)
        if dz[best] < 0.0005:  # within 0.0005 in z
            overlap_idx_pp.append(best)
            overlap_idx_des.append(i)
    
    N_overlap = len(overlap_idx_pp)
    print("\n  Redshift-matched overlap (dz < 0.0005): %d SNe" % N_overlap)
    
    # Survey composition
    if 'IDSURVEY' in pp:
        surveys_pp = pp['IDSURVEY']
        print("\n  Pantheon+ survey IDs: %s" % 
              {int(s): int(np.sum(surveys_pp == s)) for s in np.unique(surveys_pp)[:10]})
    if 'IDSURVEY' in des:
        surveys_des = des['IDSURVEY']
        print("  DES survey IDs: %s" % 
              {int(s): int(np.sum(surveys_des == s)) for s in np.unique(surveys_des)[:10]})
    
    # ── Recompute with overlap removed ──
    print("\n  Recomputing Pantheon+ ΔΩm excluding overlapping SNe...")
    
    # Quality cuts
    pp_mass = pp['HOST_LOGMASS']
    pp_good = (pp_mass > 5) & (pp_mass < 14) & (pp['zHD'] > 0.01) & np.isfinite(pp['MU_SH0ES'])
    
    # Remove overlap from Pantheon+
    pp_no_overlap = pp_good.copy()
    for idx in overlap_idx_pp:
        pp_no_overlap[idx] = False
    
    ig_full = np.where(pp_good)[0]
    ig_no_ov = np.where(pp_no_overlap)[0]
    
    print("  Pantheon+ after cuts: %d (full), %d (no overlap)" % 
          (len(ig_full), len(ig_no_ov)))
    
    def fit_Om_marg(zs, mus, Cs, label=""):
        Ci = np.linalg.inv(Cs)
        ones = np.ones(len(zs))
        def neg2logL(Om):
            if Om < 0.01 or Om > 0.99: return 1e10
            mu_th = mu_lcdm(zs, Om)
            delta = mu_th - mus
            chit2 = delta @ Ci @ delta
            B = delta @ Ci @ ones
            Csum = ones @ Ci @ ones
            return chit2 - B**2/Csum + np.log(Csum/(2*np.pi))
        from scipy.optimize import minimize_scalar
        r = minimize_scalar(neg2logL, bounds=(0.05, 0.95), method='bounded')
        Om = r.x
        h = 1e-4
        d2 = (neg2logL(Om+h) + neg2logL(Om-h) - 2*r.fun)/h**2
        sOm = np.sqrt(2.0/max(d2, 1e-10)) if d2 > 0 else 0.1
        print("    %-35s: Om=%.4f+/-%.4f" % (label, Om, sOm))
        return Om, sOm
    
    # Fit with full sample
    z_f = pp['zHD'][ig_full]; mu_f = pp['MU_SH0ES'][ig_full]; m_f = pp_mass[ig_full]
    C_f = C_pp[np.ix_(ig_full, ig_full)]
    low_f = m_f < 10; high_f = m_f >= 10
    il_f = np.where(low_f)[0]; ih_f = np.where(high_f)[0]
    
    print("\n  --- Full Pantheon+ ---")
    Om_l_f, sOm_l_f = fit_Om_marg(z_f[low_f], mu_f[low_f], C_f[np.ix_(il_f,il_f)], "Low-mass (full)")
    Om_h_f, sOm_h_f = fit_Om_marg(z_f[high_f], mu_f[high_f], C_f[np.ix_(ih_f,ih_f)], "High-mass (full)")
    dOm_f = Om_h_f - Om_l_f
    dOm_err_f = np.sqrt(sOm_h_f**2 + sOm_l_f**2)
    print("    ΔΩm = %.4f ± %.4f (%.1fσ)" % (dOm_f, dOm_err_f, abs(dOm_f)/dOm_err_f))
    
    # Fit without overlap
    z_n = pp['zHD'][ig_no_ov]; mu_n = pp['MU_SH0ES'][ig_no_ov]; m_n = pp_mass[ig_no_ov]
    C_n = C_pp[np.ix_(ig_no_ov, ig_no_ov)]
    low_n = m_n < 10; high_n = m_n >= 10
    il_n = np.where(low_n)[0]; ih_n = np.where(high_n)[0]
    
    print("\n  --- Pantheon+ without overlap ---")
    Om_l_n, sOm_l_n = fit_Om_marg(z_n[low_n], mu_n[low_n], C_n[np.ix_(il_n,il_n)], "Low-mass (no overlap)")
    Om_h_n, sOm_h_n = fit_Om_marg(z_n[high_n], mu_n[high_n], C_n[np.ix_(ih_n,ih_n)], "High-mass (no overlap)")
    dOm_n = Om_h_n - Om_l_n
    dOm_err_n = np.sqrt(sOm_h_n**2 + sOm_l_n**2)
    print("    ΔΩm = %.4f ± %.4f (%.1fσ)" % (dOm_n, dOm_err_n, abs(dOm_n)/dOm_err_n))
    
    # Combined with DES (using DES full-cov result)
    des_dOm = -0.047; des_dOm_err = 0.037
    
    for label, dOm, dOm_err in [("Full PP + DES", dOm_f, dOm_err_f),
                                  ("No-overlap PP + DES", dOm_n, dOm_err_n)]:
        w_p = 1/dOm_err**2; w_d = 1/des_dOm_err**2
        comb = (dOm*w_p + des_dOm*w_d)/(w_p + w_d)
        comb_e = 1/np.sqrt(w_p + w_d)
        print("    Combined (%s): %.4f ± %.4f (%.1fσ)" % (label, comb, comb_e, abs(comb)/comb_e))
    
    results = {
        'N_overlap_by_name': len(name_overlap),
        'N_overlap_by_z': N_overlap,
        'overlap_names': sorted(list(name_overlap))[:50],
        'full': {'delta_Om': float(dOm_f), 'err': float(dOm_err_f), 
                 'sig': float(abs(dOm_f)/dOm_err_f)},
        'no_overlap': {'delta_Om': float(dOm_n), 'err': float(dOm_err_n),
                       'sig': float(abs(dOm_n)/dOm_err_n)},
    }
    
    json.dump(results, open(os.path.join(OUTDIR, 'test1_overlap.json'), 'w'), indent=2)
    print("\n  Saved test1_overlap.json")
    return results


# ══════════════════════════════════════════════════════════════
# TEST 2: Single-likelihood γ₁z fit
# ══════════════════════════════════════════════════════════════

def test2_gamma1z():
    print("\n" + "="*70)
    print("TEST 2: Single-Likelihood z-Dependent Mass Step")
    print("="*70)
    print("  Model: μ = μ_ΛCDM(z; Ωm) + M + (γ₀ + γ₁z)/2 × tanh[(logM* - 10)/w]")
    print("  Null hypothesis: γ₁ = 0 (constant mass step)")
    print("  Alternative: γ₁ ≠ 0 (z-dependent mass step = kernel evolution)")
    
    pp, C_pp = read_pantheon()
    
    # Quality cuts
    mass = pp['HOST_LOGMASS']
    good = (mass > 5) & (mass < 14) & (pp['zHD'] > 0.01) & np.isfinite(pp['MU_SH0ES'])
    ig = np.where(good)[0]
    
    z = pp['zHD'][ig]
    mu = pp['MU_SH0ES'][ig]
    logM = mass[ig]
    C = C_pp[np.ix_(ig, ig)]
    Cinv = np.linalg.inv(C)
    N = len(z)
    ones = np.ones(N)
    
    print("  N = %d SNe after cuts" % N)
    
    # Pre-compute μ_ΛCDM on a grid for speed
    print("  Pre-computing μ_ΛCDM grid...", flush=True)
    Om_grid = np.arange(0.10, 0.60, 0.005)
    mu_grid = {}
    for Om in Om_grid:
        mu_grid[round(Om, 3)] = mu_lcdm(z, Om)
    
    def mu_lcdm_interp(Om):
        Om_r = round(min(max(Om, 0.10), 0.595), 3)
        if Om_r in mu_grid:
            return mu_grid[Om_r]
        # Fallback: compute directly
        return mu_lcdm(z, Om)
    
    # Step function
    def mass_step(logM_arr, z_arr, gamma0, gamma1, w=0.3):
        """Continuous sigmoid mass step: (γ₀ + γ₁z)/2 × tanh[(logM* - 10)/w]"""
        return 0.5 * (gamma0 + gamma1 * z_arr) * np.tanh((logM_arr - 10.0) / w)
    
    # Neg-2-log-likelihood with analytic M marginalization
    def neg2logL(params):
        Om, gamma0, gamma1 = params
        if Om < 0.05 or Om > 0.95: return 1e10
        mu_th = mu_lcdm_interp(Om)
        step = mass_step(logM, z, gamma0, gamma1)
        delta = mu_th + step - mu
        # Analytic marginalization over M (Goliath+2001)
        chit2 = delta @ Cinv @ delta
        B = delta @ Cinv @ ones
        Csum = ones @ Cinv @ ones
        return chit2 - B**2/Csum + np.log(Csum/(2*np.pi))
    
    # ── Fit 1: Null model (γ₁ = 0, only γ₀) ──
    print("\n  Fitting null model: γ₁ = 0 (constant mass step)...", flush=True)
    best_null = None
    for Om0 in [0.25, 0.30, 0.35]:
        for g0 in [-0.10, -0.05, 0.0, 0.05, 0.10]:
            try:
                r = minimize(lambda p: neg2logL([p[0], p[1], 0.0]),
                           [Om0, g0], method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-6})
                if best_null is None or r.fun < best_null.fun:
                    best_null = r
            except: pass
    
    Om_null, g0_null = best_null.x
    chi2_null = best_null.fun
    print("    Null: Ωm=%.4f, γ₀=%.4f, χ²=%.2f" % (Om_null, g0_null, chi2_null))
    
    # ── Fit 2: Alternative model (γ₁ free) ──
    print("  Fitting alternative model: γ₁ free...", flush=True)
    best_alt = None
    for Om0 in [0.25, 0.30, 0.35]:
        for g0 in [-0.05, 0.0, 0.05]:
            for g1 in [-0.15, -0.08, 0.0, 0.08, 0.15]:
                try:
                    r = minimize(neg2logL, [Om0, g0, g1], method='Nelder-Mead',
                               options={'maxiter': 10000, 'xatol': 1e-6})
                    if best_alt is None or r.fun < best_alt.fun:
                        best_alt = r
                except: pass
    
    Om_alt, g0_alt, g1_alt = best_alt.x
    chi2_alt = best_alt.fun
    print("    Alt:  Ωm=%.4f, γ₀=%.4f, γ₁=%.4f, χ²=%.2f" % 
          (Om_alt, g0_alt, g1_alt, chi2_alt))
    
    # ── Likelihood ratio test ──
    delta_chi2 = chi2_null - chi2_alt  # should be positive if γ₁ helps
    # 1 extra parameter → Δχ² ~ χ²(1 dof) under null
    from scipy.stats import chi2 as chi2_dist
    p_value = chi2_dist.sf(max(delta_chi2, 0), 1)
    n_sigma = np.sqrt(max(delta_chi2, 0))  # approximate
    
    print("\n  ── LIKELIHOOD RATIO TEST ──")
    print("    Δχ² = %.3f (1 dof)" % delta_chi2)
    print("    p-value = %.4f" % p_value)
    print("    Significance: %.1fσ" % n_sigma)
    
    # ── Error on γ₁ from Hessian ──
    h = 1e-3
    chi2_center = neg2logL([Om_alt, g0_alt, g1_alt])
    chi2_plus = neg2logL([Om_alt, g0_alt, g1_alt + h])
    chi2_minus = neg2logL([Om_alt, g0_alt, g1_alt - h])
    d2_g1 = (chi2_plus + chi2_minus - 2*chi2_center) / h**2
    sigma_g1 = np.sqrt(2.0/max(d2_g1, 1e-10)) if d2_g1 > 0 else 0.1
    
    print("\n    γ₁ = %.4f ± %.4f" % (g1_alt, sigma_g1))
    print("    γ₁/σ(γ₁) = %.1fσ" % (abs(g1_alt)/sigma_g1))
    
    # ── Interpretation ──
    print("\n  ── INTERPRETATION ──")
    if delta_chi2 > 4:
        print("    γ₁ ≠ 0 at >2σ: the mass step is z-dependent.")
        print("    This is direct evidence for the kernel evolution effect.")
    elif delta_chi2 > 1:
        print("    γ₁ ≠ 0 at 1–2σ: suggestive but not conclusive.")
    else:
        print("    No significant evidence for z-dependent mass step in this dataset.")
    
    # ── Also fit with different sigmoid widths for robustness ──
    print("\n  ── ROBUSTNESS: varying sigmoid width w ──")
    for w_test in [0.2, 0.3, 0.5, 1.0]:
        def neg2logL_w(params):
            Om, gamma0, gamma1 = params
            if Om < 0.05 or Om > 0.95: return 1e10
            mu_th = mu_lcdm_interp(Om)
            step = 0.5 * (gamma0 + gamma1 * z) * np.tanh((logM - 10.0) / w_test)
            delta = mu_th + step - mu
            chit2 = delta @ Cinv @ delta
            B = delta @ Cinv @ ones
            Csum = ones @ Cinv @ ones
            return chit2 - B**2/Csum + np.log(Csum/(2*np.pi))
        
        # Null
        r0 = minimize(lambda p: neg2logL_w([p[0], p[1], 0.0]),
                      [Om_null, g0_null], method='Nelder-Mead', 
                      options={'maxiter': 5000})
        # Alt
        r1 = minimize(neg2logL_w, [Om_alt, g0_alt, g1_alt], method='Nelder-Mead',
                      options={'maxiter': 5000})
        dchi2_w = r0.fun - r1.fun
        print("    w=%.1f: Δχ²=%.2f (%.1fσ), γ₁=%.4f" % 
              (w_test, dchi2_w, np.sqrt(max(dchi2_w, 0)), r1.x[2]))
    
    # ── Mass threshold variation ──
    print("\n  ── ROBUSTNESS: varying mass threshold ──")
    for M_cut in [9.5, 10.0, 10.5, 11.0]:
        def neg2logL_cut(params):
            Om, gamma0, gamma1 = params
            if Om < 0.05 or Om > 0.95: return 1e10
            mu_th = mu_lcdm_interp(Om)
            step = 0.5 * (gamma0 + gamma1 * z) * np.tanh((logM - M_cut) / 0.3)
            delta = mu_th + step - mu
            chit2 = delta @ Cinv @ delta
            B = delta @ Cinv @ ones
            Csum = ones @ Cinv @ ones
            return chit2 - B**2/Csum + np.log(Csum/(2*np.pi))
        
        r0 = minimize(lambda p: neg2logL_cut([p[0], p[1], 0.0]),
                      [0.3, 0.05], method='Nelder-Mead', options={'maxiter': 5000})
        r1 = minimize(neg2logL_cut, [0.3, 0.05, -0.05], method='Nelder-Mead',
                      options={'maxiter': 5000})
        dchi2_c = r0.fun - r1.fun
        print("    logM*=%.1f: Δχ²=%.2f (%.1fσ), γ₁=%.4f" % 
              (M_cut, dchi2_c, np.sqrt(max(dchi2_c, 0)), r1.x[2]))
    
    # Save
    results = {
        'null': {'Om': float(Om_null), 'gamma0': float(g0_null), 'chi2': float(chi2_null)},
        'alt': {'Om': float(Om_alt), 'gamma0': float(g0_alt), 'gamma1': float(g1_alt),
                'gamma1_err': float(sigma_g1), 'chi2': float(chi2_alt)},
        'delta_chi2': float(delta_chi2),
        'p_value': float(p_value),
        'significance_sigma': float(n_sigma),
        'gamma1_significance': float(abs(g1_alt)/sigma_g1),
    }
    
    json.dump(results, open(os.path.join(OUTDIR, 'test2_gamma1z.json'), 'w'), indent=2)
    print("\n  Saved test2_gamma1z.json")
    return results


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Paper 7: Critical Referee Tests")
    print("="*70)
    
    try:
        r1 = test1_overlap()
    except Exception as e:
        print("  Test 1 failed: %s" % e)
        import traceback; traceback.print_exc()
        r1 = None
    
    try:
        r2 = test2_gamma1z()
    except Exception as e:
        print("  Test 2 failed: %s" % e)
        import traceback; traceback.print_exc()
        r2 = None
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    if r1:
        print("  Test 1 (overlap): %d overlapping SNe by name, %d by z-match" % 
              (r1['N_overlap_by_name'], r1['N_overlap_by_z']))
        print("    Full PP:       ΔΩm = %.4f ± %.4f (%.1fσ)" % 
              (r1['full']['delta_Om'], r1['full']['err'], r1['full']['sig']))
        print("    No-overlap PP: ΔΩm = %.4f ± %.4f (%.1fσ)" % 
              (r1['no_overlap']['delta_Om'], r1['no_overlap']['err'], r1['no_overlap']['sig']))
    if r2:
        print("  Test 2 (γ₁z): γ₁ = %.4f ± %.4f (%.1fσ)" % 
              (r2['alt']['gamma1'], r2['alt']['gamma1_err'], r2['gamma1_significance']))
        print("    Δχ² = %.2f → %.1fσ evidence for z-dependent mass step" % 
              (r2['delta_chi2'], r2['significance_sigma']))
