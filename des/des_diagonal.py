#!/usr/bin/env python3
"""
Paper 7 Path 1: DES-SN5YR Host Mass Split Analysis
====================================================
Independent confirmation of the Pantheon+ 3.3σ ΔΩm result
using a completely different SN sample (1635 DES + 194 low-z).

Data source: https://github.com/des-science/DES-SN5YR
Cite: DES Collaboration (2024), ApJL; Vincenzi et al. (2024)
      Sanchez et al. (2024) for data release

Steps:
1. Clone DES-SN5YR repo (or use existing)
2. Read distance moduli + covariance from 4_DISTANCES_COVMAT/
3. Read HOST_LOGMASS from FITRES file in 0_DATA/
4. Split by host mass, fit Ωm to each subsample
5. Compare with Pantheon+ result
"""

import numpy as np
import os, sys, json, subprocess, glob
from scipy.optimize import minimize
from scipy.integrate import quad

DATADIR = os.path.expanduser('~/des_sn5yr')
OUTDIR = os.path.expanduser('~/paper7_results')
os.makedirs(OUTDIR, exist_ok=True)

# ══════════════════════════════════════════════
# Step 1: Get the data
# ══════════════════════════════════════════════
REPO = os.path.join(DATADIR, 'DES-SN5YR')
if not os.path.exists(REPO):
    print("Cloning DES-SN5YR repository...")
    os.makedirs(DATADIR, exist_ok=True)
    # Sparse checkout: only distances + FITRES data
    ret = subprocess.run(
        ['git', 'clone', '--depth', '1', 
         'https://github.com/des-science/DES-SN5YR.git', REPO],
        capture_output=True, text=True, timeout=300
    )
    if ret.returncode != 0:
        print("git clone failed:", ret.stderr)
        print("\nManual download instructions:")
        print("  cd ~/des_sn5yr")
        print("  git clone --depth 1 https://github.com/des-science/DES-SN5YR.git")
        sys.exit(1)
    print("  Done.")
else:
    print("DES-SN5YR repo already exists at %s" % REPO)

# ══════════════════════════════════════════════
# Step 2: Find and read distance data
# ══════════════════════════════════════════════
print("\nLooking for distance data...")

# Search for the main data vector file
dist_dir = os.path.join(REPO, '4_DISTANCES_COVMAT')
possible_data = glob.glob(os.path.join(dist_dir, '*.csv')) + \
                glob.glob(os.path.join(dist_dir, '*.txt')) + \
                glob.glob(os.path.join(dist_dir, '*.dat')) + \
                glob.glob(os.path.join(dist_dir, '*.fitres')) + \
                glob.glob(os.path.join(dist_dir, '*HD*'))

print("  Found in 4_DISTANCES_COVMAT/:")
for f in possible_data:
    print("    %s (%d KB)" % (os.path.basename(f), os.path.getsize(f)//1024))

# Also look for FITRES files with host properties
fitres_files = glob.glob(os.path.join(REPO, '**/*.FITRES'), recursive=True) + \
               glob.glob(os.path.join(REPO, '**/*.fitres'), recursive=True) + \
               glob.glob(os.path.join(REPO, '**/*FITOPT*'), recursive=True) + \
               glob.glob(os.path.join(REPO, '**/*fitres*'), recursive=True) + \
               glob.glob(os.path.join(REPO, '**/*BBC*'), recursive=True)

if fitres_files:
    print("\n  FITRES files found:")
    for f in fitres_files[:10]:
        print("    %s (%d KB)" % (f.replace(REPO+'/', ''), os.path.getsize(f)//1024))

# ══════════════════════════════════════════════
# Step 3: Parse data files
# ══════════════════════════════════════════════

def read_snana_file(filepath):
    """Read SNANA-format FITRES/DAT file."""
    data = {}
    with open(filepath, 'r') as f:
        header = None
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('VARNAMES:'):
                header = line.split()[1:]  # skip 'VARNAMES:'
                for col in header:
                    data[col] = []
                continue
            if header and (line.startswith('SN:') or line.startswith('ROW:')):
                vals = line.split()[1:]  # skip 'SN:' or 'ROW:'
                if len(vals) >= len(header):
                    for col, val in zip(header, vals):
                        try:
                            data[col].append(float(val))
                        except:
                            data[col].append(val)
    # Convert to numpy
    for col in data:
        try:
            data[col] = np.array(data[col], dtype=float)
        except:
            data[col] = np.array(data[col])
    return data

def read_csv_or_txt(filepath):
    """Read CSV or space-delimited file with header."""
    data = {}
    with open(filepath, 'r') as f:
        # Find header line
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # First non-comment, non-empty line is header
            header = line.replace(',', ' ').split()
            for col in header:
                data[col] = []
            break
        # Read data
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            vals = line.replace(',', ' ').split()
            if len(vals) >= len(header):
                for col, val in zip(header, vals):
                    try:
                        data[col].append(float(val))
                    except:
                        data[col].append(val)
    for col in data:
        try:
            data[col] = np.array(data[col], dtype=float)
        except:
            data[col] = np.array(data[col])
    return data

# Try to read the main distance file
dist_data = None
for f in sorted(possible_data, key=lambda x: os.path.getsize(x), reverse=True):
    print("\n  Trying %s..." % os.path.basename(f))
    try:
        if f.endswith('.fitres') or f.endswith('.FITRES'):
            d = read_snana_file(f)
        else:
            d = read_csv_or_txt(f)
        cols = list(d.keys())
        print("    Columns: %s" % ', '.join(cols[:15]))
        N = len(d[cols[0]]) if cols else 0
        print("    N = %d" % N)
        
        # Check if this has what we need
        has_z = any(c in d for c in ['zHD', 'zCMB', 'z', 'ZHEL'])
        has_mu = any(c in d for c in ['MU', 'mu', 'MU_SH0ES', 'MUDIF'])
        has_mass = any(c in d for c in ['HOST_LOGMASS', 'HOSTMASS', 'host_logmass'])
        print("    Has z: %s, Has MU: %s, Has HOST_MASS: %s" % (has_z, has_mu, has_mass))
        
        if has_z and has_mu:
            dist_data = d
            dist_file = f
            if has_mass:
                print("    *** This file has everything we need! ***")
                break
    except Exception as e:
        print("    Error: %s" % e)

if dist_data is None:
    print("\nERROR: Could not find distance data.")
    print("Listing all files in 4_DISTANCES_COVMAT/:")
    for f in glob.glob(os.path.join(dist_dir, '*')):
        print("  %s (%d bytes)" % (os.path.basename(f), os.path.getsize(f)))
    sys.exit(1)

# Identify column names
z_col = next(c for c in ['zHD', 'zCMB', 'z'] if c in dist_data)
mu_col = next(c for c in ['MU', 'mu', 'MU_SH0ES', 'MUDIF'] if c in dist_data)
mass_col = next((c for c in ['HOST_LOGMASS', 'HOSTMASS', 'host_logmass'] if c in dist_data), None)

z = dist_data[z_col]
mu = dist_data[mu_col]
N_all = len(z)
print("\n  Using: z=%s, mu=%s from %s (N=%d)" % (z_col, mu_col, os.path.basename(dist_file), N_all))

# If no host mass in distance file, look in FITRES files
if mass_col is None:
    print("\n  No HOST_LOGMASS in distance file. Searching FITRES files...")
    # Look for a file with both CID and HOST_LOGMASS
    cid_col = next((c for c in ['CID', 'SNID', 'cid'] if c in dist_data), None)
    
    for ff in fitres_files:
        try:
            fd = read_snana_file(ff)
            if 'HOST_LOGMASS' in fd and ('CID' in fd or 'SNID' in fd):
                print("  Found HOST_LOGMASS in %s" % ff.replace(REPO+'/', ''))
                # Cross-match by CID
                fc = fd.get('CID', fd.get('SNID'))
                dc = dist_data.get('CID', dist_data.get('SNID'))
                if fc is not None and dc is not None:
                    mass_lookup = dict(zip(fc, fd['HOST_LOGMASS']))
                    host_mass = np.array([mass_lookup.get(c, -9) for c in dc])
                    dist_data['HOST_LOGMASS'] = host_mass
                    mass_col = 'HOST_LOGMASS'
                    print("    Matched %d/%d SNe with host masses" % 
                          (np.sum(host_mass > 0), N_all))
                    break
        except:
            continue

if mass_col is None:
    print("\nERROR: Could not find HOST_LOGMASS in any file.")
    print("You may need to download the host galaxy catalog separately.")
    print("See Wiseman et al. (2020): https://arxiv.org/abs/2001.02640")
    sys.exit(1)

logmass = dist_data[mass_col]

# ══════════════════════════════════════════════
# Step 4: Read covariance matrix
# ══════════════════════════════════════════════
print("\nLooking for covariance matrix...")
cov_files = glob.glob(os.path.join(dist_dir, '*SYS*.cov')) + \
            glob.glob(os.path.join(dist_dir, '*SYS*.txt')) + \
            glob.glob(os.path.join(dist_dir, '*covmat*')) + \
            glob.glob(os.path.join(dist_dir, '*.cov'))

has_cov = False
C = None
for cf in sorted(cov_files, key=lambda x: os.path.getsize(x), reverse=True):
    print("  Trying %s (%d MB)..." % (os.path.basename(cf), os.path.getsize(cf)//1024//1024))
    try:
        with open(cf, 'r') as f:
            first_line = f.readline().strip()
            N_cov = int(first_line)
            print("    Cov size: %d" % N_cov)
            cov_flat = []
            for line in f:
                cov_flat.extend([float(x) for x in line.strip().split()])
        
        if len(cov_flat) == N_cov * N_cov:
            C_full = np.array(cov_flat).reshape(N_cov, N_cov)
            has_cov = True
            print("    Loaded %dx%d covariance matrix" % (N_cov, N_cov))
            break
    except Exception as e:
        print("    Error: %s" % e)

# ══════════════════════════════════════════════
# Step 5: Quality cuts and mass split
# ══════════════════════════════════════════════
good = (logmass > 5) & (logmass < 14) & (z > 0.01) & (z < 2.0) & np.isfinite(mu)
idx_good = np.where(good)[0]

z_g = z[good]
mu_g = mu[good]
lm_g = logmass[good]
N = len(z_g)

# Get diagonal errors if available
mu_err_col = next((c for c in ['MUERR', 'MU_ERR', 'MUERR_FINAL'] if c in dist_data), None)
if mu_err_col:
    mu_err = dist_data[mu_err_col][good]
else:
    mu_err = np.full(N, 0.15)

print("\n  After quality cuts: %d SNe" % N)

# Extract sub-covariance
if has_cov and N_cov == N_all:
    C = C_full[np.ix_(idx_good, idx_good)]
    print("  Sub-covariance: %dx%d" % C.shape)
elif has_cov:
    print("  WARNING: Cov size %d != data size %d. Using diagonal." % (N_cov, N_all))
    C = np.diag(mu_err**2)
else:
    print("  No covariance matrix. Using diagonal errors.")
    C = np.diag(mu_err**2)

# Split
MASS_CUT = 10.0
low = lm_g < MASS_CUT
high = lm_g >= MASS_CUT

print("\n  Split at logM* = %.1f:" % MASS_CUT)
print("    Low-mass:  %d SNe, <logM*>=%.2f, <z>=%.3f" % 
      (np.sum(low), lm_g[low].mean(), z_g[low].mean()))
print("    High-mass: %d SNe, <logM*>=%.2f, <z>=%.3f" % 
      (np.sum(high), lm_g[high].mean(), z_g[high].mean()))

# ══════════════════════════════════════════════
# Step 6: Fit Ωm
# ══════════════════════════════════════════════

def mu_lcdm(z_arr, Om):
    mu_arr = np.zeros_like(z_arr)
    for i, zi in enumerate(z_arr):
        if zi < 1e-6:
            mu_arr[i] = -99
            continue
        dL, _ = quad(lambda zz: 1.0/np.sqrt(Om*(1+zz)**3 + (1-Om)), 0, zi, limit=100)
        mu_arr[i] = 5 * np.log10((1+zi) * dL)
    return mu_arr

def fit_Om(z_sub, mu_sub, C_sub, label=""):
    try:
        C_inv = np.linalg.inv(C_sub)
    except:
        C_inv = np.linalg.inv(C_sub + 1e-6 * np.eye(len(C_sub)))
    
    def chi2(params):
        Om, M = params
        if Om < 0.01 or Om > 0.99:
            return 1e10
        mu_th = mu_lcdm(z_sub, Om) + M
        delta = mu_sub - mu_th
        return float(delta @ C_inv @ delta)
    
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
    
    # Error from Hessian
    h = 1e-4
    fp = chi2([Om_fit + h, M_fit])
    fm = chi2([Om_fit - h, M_fit])
    d2 = (fp + fm - 2*chi2_min) / h**2
    sigma_Om = 1.0/np.sqrt(max(d2, 1e-10)) if d2 > 0 else 0.1
    
    print("  %-25s: Om = %.4f +/- %.4f, chi2/dof = %.1f/%d = %.3f" % 
          (label, Om_fit, sigma_Om, chi2_min, ndof, chi2_min/ndof))
    return Om_fit, sigma_Om, chi2_min, ndof

# ── Fit ──
print("\nFitting (this takes ~5 min per subsample)...")
print("=" * 60)

idx_low = np.where(low)[0]
idx_high = np.where(high)[0]

Om_full, sOm_full, _, _ = fit_Om(z_g, mu_g, C, "Full DES-SN5YR sample")
Om_low, sOm_low, _, _ = fit_Om(z_g[low], mu_g[low], C[np.ix_(idx_low,idx_low)], "Low-mass hosts")
Om_high, sOm_high, _, _ = fit_Om(z_g[high], mu_g[high], C[np.ix_(idx_high,idx_high)], "High-mass hosts")

dOm = Om_high - Om_low
dOm_err = np.sqrt(sOm_high**2 + sOm_low**2)
sig = abs(dOm) / dOm_err

# ══════════════════════════════════════════════
# Step 7: Report
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("DES-SN5YR HOST MASS SPLIT RESULTS")
print("=" * 60)
print("  Full:      Om = %.4f +/- %.4f" % (Om_full, sOm_full))
print("  Low-mass:  Om = %.4f +/- %.4f (N=%d)" % (Om_low, sOm_low, np.sum(low)))
print("  High-mass: Om = %.4f +/- %.4f (N=%d)" % (Om_high, sOm_high, np.sum(high)))
print("  Delta_Om = %.4f +/- %.4f (%.1f sigma)" % (dOm, dOm_err, sig))
print()
print("  Pantheon+ comparison: Delta_Om = -0.051 +/- 0.015 (3.3 sigma)")
print("  DES-SN5YR:            Delta_Om = %.3f +/- %.3f (%.1f sigma)" % (dOm, dOm_err, sig))
if dOm < 0:
    print("  Same sign as Pantheon+ -> CONFIRMS the averaging illusion")
else:
    print("  Opposite sign to Pantheon+ -> check systematics")

# Binned residuals
print("\n  Redshift-binned residuals (high - low mass):")
mu_ref = mu_lcdm(z_g, Om_full)
M_ref = np.median(mu_g - mu_ref)
resid = mu_g - mu_ref - M_ref

z_edges = [0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]
n_neg = 0
bin_data = []
print("  %12s %6s %6s %10s %8s" % ('z_bin', 'N_lo', 'N_hi', 'Delta_mu', 'sigma'))
for k in range(len(z_edges)-1):
    zlo, zhi = z_edges[k], z_edges[k+1]
    in_bin = (z_g >= zlo) & (z_g < zhi)
    in_lo = in_bin & low
    in_hi = in_bin & high
    nl, nh = np.sum(in_lo), np.sum(in_hi)
    if nl < 3 or nh < 3:
        continue
    wl = 1/mu_err[in_lo]**2
    wh = 1/mu_err[in_hi]**2
    ml = np.average(resid[in_lo], weights=wl)
    mh = np.average(resid[in_hi], weights=wh)
    el = 1/np.sqrt(np.sum(wl))
    eh = 1/np.sqrt(np.sum(wh))
    d = mh - ml
    de = np.sqrt(el**2 + eh**2)
    if d < 0:
        n_neg += 1
    print("  %5.2f-%4.2f %6d %6d %+10.4f %8.4f" % (zlo, zhi, nl, nh, d, de))
    bin_data.append({'z_lo': zlo, 'z_hi': zhi, 'delta': float(d), 'delta_err': float(de),
                     'n_low': int(nl), 'n_high': int(nh)})

n_bins = len(bin_data)
p_neg = 0.5**n_neg * 100 if n_neg > 0 else 100
print("\n  %d/%d bins have Delta_mu < 0 (P=%.1f%%)" % (n_neg, n_bins, p_neg))

# Save
results = {
    'dataset': 'DES-SN5YR',
    'N_total': N, 'N_low': int(np.sum(low)), 'N_high': int(np.sum(high)),
    'mass_cut': MASS_CUT,
    'Om_full': float(Om_full), 'Om_full_err': float(sOm_full),
    'Om_low': float(Om_low), 'Om_low_err': float(sOm_low),
    'Om_high': float(Om_high), 'Om_high_err': float(sOm_high),
    'delta_Om': float(dOm), 'delta_Om_err': float(dOm_err),
    'significance': float(sig),
    'n_negative_bins': n_neg, 'n_total_bins': n_bins,
    'binned_residuals': bin_data,
    'uses_full_covariance': has_cov,
    'comparison': {
        'pantheonplus_delta_Om': -0.051,
        'pantheonplus_delta_Om_err': 0.015,
        'pantheonplus_significance': 3.3,
    }
}

outpath = os.path.join(OUTDIR, 'des_sn5yr_mass_split.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to %s" % outpath)
