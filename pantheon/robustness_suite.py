#!/usr/bin/env python3
"""
Paper 7: Tier 1-2 Improvements
================================
1a. Redshift-reweighted Pantheon+ split
1b. Mass threshold scan (logM* = 9.0 to 11.0)
1c. Low-z-only vs high-z-only split
2.  THESAN bootstrap uncertainties on α_z
5.  Cross-covariance σ(ΔΩm) from off-diagonal block
"""

import numpy as np
import os, json, sys
from scipy.optimize import minimize_scalar, minimize
from scipy.integrate import quad

PANTHEON_DIR = os.path.expanduser('~/pantheonplus')
THESAN_DIR = os.path.expanduser('~/thesan_ion')  # adjust if different
OUTDIR = os.path.expanduser('~/paper7_results')
os.makedirs(OUTDIR, exist_ok=True)

def read_plain(fpath):
    with open(fpath) as f:
        header = f.readline().strip().split()
        data = {c: [] for c in header}
        for line in f:
            vals = line.strip().split()
            if len(vals) < len(header): continue
            for c, v in zip(header, vals):
                try: data[c].append(float(v))
                except: data[c].append(v)
    for c in data:
        try: data[c] = np.array(data[c], dtype=float)
        except: data[c] = np.array(data[c])
    return data

def mu_lcdm(z_arr, Om):
    out = np.zeros_like(z_arr)
    for i, zi in enumerate(z_arr):
        if zi < 1e-6: out[i] = -99; continue
        dL, _ = quad(lambda zz: 1.0/np.sqrt(Om*(1+zz)**3+(1-Om)), 0, zi, limit=100)
        out[i] = 5*np.log10((1+zi)*dL)
    return out

# ══════════════════════════════════════════════════
# Load Pantheon+
# ══════════════════════════════════════════════════
print("Loading Pantheon+...", flush=True)
pp = read_plain(os.path.join(PANTHEON_DIR, 'Pantheon+SH0ES.dat'))
with open(os.path.join(PANTHEON_DIR, 'Pantheon+SH0ES_STAT+SYS.cov')) as f:
    Npp = int(f.readline().strip())
    vals = []
    for line in f:
        vals.extend([float(x) for x in line.strip().split()])
C_pp = np.array(vals).reshape(Npp, Npp)

z_all = pp['zHD']; mu_all = pp['MU_SH0ES']; mass_all = pp['HOST_LOGMASS']
good = (mass_all > 5) & (mass_all < 14) & (z_all > 0.01) & np.isfinite(mu_all)
ig = np.where(good)[0]
z = z_all[ig]; mu = mu_all[ig]; mass = mass_all[ig]
C = C_pp[np.ix_(ig, ig)]
Cinv = np.linalg.inv(C)
N = len(z)
ones = np.ones(N)
print("  N = %d" % N)

# Pre-compute mu grid
print("  Building μ grid...", flush=True)
mu_cache = {}
for Om in np.arange(0.10, 0.55, 0.005):
    mu_cache[round(Om, 3)] = mu_lcdm(z, Om)

def get_mu(Om):
    k = round(min(max(Om, 0.10), 0.545), 3)
    return mu_cache.get(k, mu_lcdm(z, Om))

def fit_Om_marg(zs, mus, Cs, label=""):
    Ci = np.linalg.inv(Cs)
    os1 = np.ones(len(zs))
    # Need mu_lcdm for this subset
    def neg2ll(Om):
        if Om < 0.05 or Om > 0.95: return 1e10
        mt = mu_lcdm(zs, Om)
        d = mt - mus
        c2 = d @ Ci @ d; B = d @ Ci @ os1; S = os1 @ Ci @ os1
        return c2 - B**2/S + np.log(S/(2*np.pi))
    r = minimize_scalar(neg2ll, bounds=(0.05, 0.95), method='bounded')
    h = 1e-4
    d2 = (neg2ll(r.x+h)+neg2ll(r.x-h)-2*r.fun)/h**2
    s = np.sqrt(2/max(d2, 1e-10)) if d2 > 0 else 0.1
    return r.x, s

def do_split(z_sub, mu_sub, mass_sub, C_sub, Mcut, label):
    low = mass_sub < Mcut; high = mass_sub >= Mcut
    il = np.where(low)[0]; ih = np.where(high)[0]
    if np.sum(low) < 10 or np.sum(high) < 10:
        return None, None, None
    Ol, sl = fit_Om_marg(z_sub[low], mu_sub[low], C_sub[np.ix_(il,il)])
    Oh, sh = fit_Om_marg(z_sub[high], mu_sub[high], C_sub[np.ix_(ih,ih)])
    dO = Oh - Ol; de = np.sqrt(sh**2 + sl**2)
    sig = abs(dO)/de
    print("  %-40s: dOm=%.4f+/-%.4f (%.1fσ) N_lo=%d N_hi=%d" %
          (label, dO, de, sig, np.sum(low), np.sum(high)))
    return dO, de, sig


# ══════════════════════════════════════════════════
# 1a. Redshift-reweighted
# ══════════════════════════════════════════════════
print("\n" + "="*65)
print("1a. REDSHIFT-REWEIGHTED SPLIT")
print("="*65)
print("  Reweight low-mass SNe to match high-mass z-distribution")

low = mass < 10; high = mass >= 10

# Build z-histograms
z_edges = np.linspace(0, 1.5, 16)
h_high, _ = np.histogram(z[high], z_edges, density=True)
h_low, _ = np.histogram(z[low], z_edges, density=True)

# Compute weights for low-mass sample
weights_low = np.ones(np.sum(low))
z_low = z[low]
for i, zi in enumerate(z_low):
    ibin = np.searchsorted(z_edges, zi) - 1
    ibin = max(0, min(ibin, len(h_high)-1))
    if h_low[ibin] > 0:
        weights_low[i] = h_high[ibin] / h_low[ibin]
    else:
        weights_low[i] = 0

# Effective number of reweighted SNe
n_eff = np.sum(weights_low)**2 / np.sum(weights_low**2)
print("  N_eff(low, reweighted) = %.0f (from %d)" % (n_eff, np.sum(low)))

# Fit low-mass with reweighted diagonal (approximate)
il = np.where(low)[0]; ih = np.where(high)[0]
C_low = C[np.ix_(il, il)]
W = np.diag(weights_low)
# Weighted covariance: W^{1/2} C W^{1/2} / (sum w)
# For simplicity, use diagonal reweighting
C_low_rw = np.diag(np.diag(C_low) / (weights_low**2 + 1e-30))

Ol_rw, sl_rw = fit_Om_marg(z_low, mu[low], C_low_rw)
Oh, sh = fit_Om_marg(z[high], mu[high], C[np.ix_(ih, ih)])
dO_rw = Oh - Ol_rw; de_rw = np.sqrt(sh**2 + sl_rw**2)
print("  Reweighted: dOm=%.4f+/-%.4f (%.1fσ)" % (dO_rw, de_rw, abs(dO_rw)/de_rw))
print("  Original:   dOm=%.4f" % (Oh - fit_Om_marg(z[low], mu[low], C_low)[0]))

# ══════════════════════════════════════════════════
# 1b. Mass threshold scan
# ══════════════════════════════════════════════════
print("\n" + "="*65)
print("1b. MASS THRESHOLD SCAN")
print("="*65)

threshold_results = []
for Mc in np.arange(9.0, 11.25, 0.25):
    dO, de, sig = do_split(z, mu, mass, C, Mc, "logM*=%.2f" % Mc)
    if dO is not None:
        threshold_results.append({'Mcut': float(Mc), 'dOm': float(dO), 
                                  'err': float(de), 'sig': float(sig)})

# ══════════════════════════════════════════════════
# 1c. Low-z only vs high-z only
# ══════════════════════════════════════════════════
print("\n" + "="*65)
print("1c. LOW-z vs HIGH-z SPLIT")
print("="*65)

for z_split in [0.05, 0.08, 0.10, 0.15, 0.20]:
    mask = z < z_split
    if np.sum(mask & (mass < 10)) < 10 or np.sum(mask & (mass >= 10)) < 10:
        print("  z < %.2f: too few SNe" % z_split)
        continue
    idx = np.where(mask)[0]
    do_split(z[mask], mu[mask], mass[mask], C[np.ix_(idx,idx)], 10.0,
             "z < %.2f only" % z_split)

for z_split in [0.05, 0.08, 0.10, 0.15, 0.20]:
    mask = z >= z_split
    if np.sum(mask & (mass < 10)) < 10 or np.sum(mask & (mass >= 10)) < 10:
        print("  z >= %.2f: too few SNe" % z_split)
        continue
    idx = np.where(mask)[0]
    do_split(z[mask], mu[mask], mass[mask], C[np.ix_(idx,idx)], 10.0,
             "z >= %.2f only" % z_split)


# ══════════════════════════════════════════════════
# 5. Cross-covariance σ(ΔΩm)
# ══════════════════════════════════════════════════
print("\n" + "="*65)
print("5. CROSS-COVARIANCE σ(ΔΩm)")
print("="*65)

# The proper error on ΔΩm = Ωm_high - Ωm_low includes the cross-covariance
# σ²(ΔΩm) = σ²(Ωm_high) + σ²(Ωm_low) - 2·Cov(Ωm_high, Ωm_low)
# 
# We estimate Cov(Ωm_high, Ωm_low) by computing dΩm/dμ for each subsample
# and propagating through the cross-covariance block C_LH.

il = np.where(mass < 10)[0]
ih = np.where(mass >= 10)[0]

# Get best-fit Ωm for each
Om_l, s_l = fit_Om_marg(z[mass<10], mu[mass<10], C[np.ix_(il,il)])
Om_h, s_h = fit_Om_marg(z[mass>=10], mu[mass>=10], C[np.ix_(ih,ih)])

# Numerical gradient dΩm/dμ_i for each subsample
# For low-mass: perturb each mu_i, refit Ωm
# This is expensive, so use Fisher approximation instead:
# σ²(Ωm) ≈ (J^T C^{-1} J)^{-1} where J_i = dμ_theory/dΩm

h_Om = 1e-4
mu_l_p = mu_lcdm(z[mass<10], Om_l + h_Om)
mu_l_m = mu_lcdm(z[mass<10], Om_l - h_Om)
J_l = (mu_l_p - mu_l_m) / (2*h_Om)  # dμ/dΩm for low-mass

mu_h_p = mu_lcdm(z[mass>=10], Om_h + h_Om)
mu_h_m = mu_lcdm(z[mass>=10], Om_h - h_Om)
J_h = (mu_h_p - mu_h_m) / (2*h_Om)

C_ll = C[np.ix_(il, il)]
C_hh = C[np.ix_(ih, ih)]
C_lh = C[np.ix_(il, ih)]  # cross-covariance block

Cinv_l = np.linalg.inv(C_ll)
Cinv_h = np.linalg.inv(C_hh)

# Fisher for each: F = J^T C^{-1} J
F_l = J_l @ Cinv_l @ J_l
F_h = J_h @ Cinv_h @ J_h

# Cross-term: dΩm_low/dμ_j × C_jk × dΩm_high/dμ_k
# = (J_l^T C_ll^{-1}) C_lh (C_hh^{-1} J_h) / (F_l * F_h)
# Actually: Cov(Ωm_l, Ωm_h) = (1/F_l) (J_l^T Cinv_l C_lh Cinv_h J_h) (1/F_h)

cross_num = J_l @ Cinv_l @ C_lh @ Cinv_h @ J_h
Cov_Om = cross_num / (F_l * F_h) * (F_l * F_h) / (F_l * F_h)
# Simpler: Cov(Ωm_l, Ωm_h) ≈ (J_l^T Cinv_l C_lh Cinv_h J_h) / (F_l * F_h)
# Wait, let me redo this properly.
# 
# For a linear model μ = μ(Ωm) + M, the Fisher estimate gives:
# (Ωm_hat - Ωm_true) ≈ (J^T C^{-1} J)^{-1} J^T C^{-1} (μ_obs - μ_true)
# So Ωm_hat_l = sum_i w_l_i * μ_i (low-mass SNe)
# Ωm_hat_h = sum_j w_h_j * μ_j (high-mass SNe)
# Cov(Ωm_l, Ωm_h) = sum_ij w_l_i * C_ij * w_h_j
#
# where w = (J^T C^{-1} J)^{-1} C^{-1} J / some normalization
# But with M marginalization this gets messy. Let's just compute numerically.

# Actually the simplest approach: the weight vector for Ωm is
# w = F^{-1} * C^{-1} J (for non-marginalized M)
# With M marginalized, it's more complex. Let's use a simple estimate.

w_l = Cinv_l @ J_l / F_l
w_h = Cinv_h @ J_h / F_h

cov_cross = w_l @ C_lh @ w_h
sigma_naive = np.sqrt(1/F_l + 1/F_h)
sigma_proper = np.sqrt(1/F_l + 1/F_h - 2*cov_cross)

dOm = Om_h - Om_l
print("  Om_low = %.4f +/- %.4f" % (Om_l, 1/np.sqrt(F_l)))
print("  Om_high = %.4f +/- %.4f" % (Om_h, 1/np.sqrt(F_h)))
print("  Cov(Om_low, Om_high) = %.6f" % cov_cross)
print("  Correlation = %.3f" % (cov_cross / (1/np.sqrt(F_l) * 1/np.sqrt(F_h))))
print()
print("  σ(ΔΩm) naive (independent):  %.4f → %.1fσ" % (sigma_naive, abs(dOm)/sigma_naive))
print("  σ(ΔΩm) proper (with cross):  %.4f → %.1fσ" % (sigma_proper, abs(dOm)/sigma_proper))
if cov_cross > 0:
    print("  → Cross-covariance is POSITIVE (shared calibration)")
    print("  → Proper error is SMALLER → significance INCREASES")
else:
    print("  → Cross-covariance is NEGATIVE")
    print("  → Proper error is LARGER → significance DECREASES")


# ══════════════════════════════════════════════════
# Save all results
# ══════════════════════════════════════════════════

results = {
    'reweighted': {'dOm': float(dO_rw), 'err': float(de_rw), 
                   'sig': float(abs(dO_rw)/de_rw)},
    'threshold_scan': threshold_results,
    'cross_covariance': {
        'cov_cross': float(cov_cross),
        'sigma_naive': float(sigma_naive),
        'sigma_proper': float(sigma_proper),
        'sig_naive': float(abs(dOm)/sigma_naive),
        'sig_proper': float(abs(dOm)/sigma_proper),
    }
}

json.dump(results, open(os.path.join(OUTDIR, 'tier12_robustness.json'), 'w'), indent=2)
print("\nSaved tier12_robustness.json")
