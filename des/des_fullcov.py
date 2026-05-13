#!/usr/bin/env python3
"""
DES-SN5YR host mass split — WITH full covariance matrix.
Covariance is in STAT+SYS.npz as upper-triangle flat array.
1,657,110 = 1820 * 1821 / 2
"""
import numpy as np
import os, sys, json
from scipy.optimize import minimize
from scipy.integrate import quad

REPO = os.path.expanduser('~/des_sn5yr/DES-SN5YR')
DIST_DIR = os.path.join(REPO, '4_DISTANCES_COVMAT')
OUTDIR = os.path.expanduser('~/paper7_results')
os.makedirs(OUTDIR, exist_ok=True)

# ── 1. Read distances from HD.csv ──
print("Reading DES-Dovekie_HD.csv...", flush=True)
hd_file = os.path.join(DIST_DIR, 'DES-Dovekie_HD.csv')
with open(hd_file, 'r') as f:
    header = None
    rows = []
    for line in f:
        line = line.strip()
        if line.startswith('#'):
            # Last comment line before data contains column names
            header_candidate = line.lstrip('#').strip()
            continue
        if header is None:
            # First non-comment line — could be header or data
            # Check if it contains non-numeric first field
            parts = line.replace(',', ' ').split()
            try:
                float(parts[0])
                # It's data — use the last comment line as header
                header = header_candidate.replace(',', ' ').split()
                rows.append(parts)
            except ValueError:
                header = line.replace(',', ' ').split()
            continue
        rows.append(line.replace(',', ' ').split())

# If header is still None, try first row
if header is None:
    print("ERROR: Could not find header")
    sys.exit(1)

print("  Columns: %s" % ', '.join(header[:10]))
print("  N rows: %d" % len(rows))

hd = {}
for i, col in enumerate(header):
    vals = []
    for row in rows:
        if i < len(row):
            try:
                vals.append(float(row[i]))
            except:
                vals.append(row[i])
        else:
            vals.append(np.nan)
    try:
        hd[col] = np.array(vals, dtype=float)
    except:
        hd[col] = np.array(vals)

# Find z and mu columns
z_col = next((c for c in ['zHD', 'zCMB', 'z'] if c in hd), None)
mu_col = next((c for c in ['MU', 'mu'] if c in hd), None)
muerr_col = next((c for c in ['MUERR', 'MU_ERR'] if c in hd), None)

print("  z=%s, mu=%s, muerr=%s" % (z_col, mu_col, muerr_col))
N_hd = len(hd[z_col])
print("  N_hd = %d" % N_hd)

# ── 2. Read metadata for HOST_LOGMASS ──
print("\nReading DES-Dovekie_Metadata.csv...", flush=True)
meta_file = os.path.join(DIST_DIR, 'DES-Dovekie_Metadata.csv')

meta = {}
with open(meta_file, 'r') as f:
    meta_header = None
    for line in f:
        line = line.strip()
        if line.startswith('VARNAMES:'):
            meta_header = line.split()[1:]
            for col in meta_header:
                meta[col] = []
            continue
        if meta_header and line.startswith('SN:'):
            vals = line.split()[1:]
            for col, val in zip(meta_header, vals):
                try:
                    meta[col].append(float(val))
                except:
                    meta[col].append(val)

for col in meta:
    try:
        meta[col] = np.array(meta[col], dtype=float)
    except:
        meta[col] = np.array(meta[col])

N_meta = len(meta[list(meta.keys())[0]])
print("  N_meta = %d" % N_meta)
print("  Columns: %s" % ', '.join(list(meta.keys())[:15]))

# Find HOST_LOGMASS
mass_col = next((c for c in ['HOST_LOGMASS', 'HOSTMASS', 'HOST_LOGMASS_ERR'] 
                 if c in meta and 'ERR' not in c), None)
if mass_col is None:
    # Search all columns for 'mass' or 'MASS'
    mass_candidates = [c for c in meta.keys() if 'MASS' in c.upper() and 'ERR' not in c.upper()]
    print("  Mass candidates: %s" % mass_candidates)
    if mass_candidates:
        mass_col = mass_candidates[0]

if mass_col:
    print("  Found mass column: %s" % mass_col)
    print("  Sample values: %s" % meta[mass_col][:5])
else:
    print("  WARNING: No HOST_LOGMASS found in metadata")
    print("  All columns: %s" % list(meta.keys()))

# ── 3. Cross-match HD and Metadata ──
# HD might not have CID — check
hd_has_cid = 'CID' in hd or 'SNID' in hd or 'cid' in hd
meta_has_cid = 'CID' in meta or 'CIDint' in meta

if hd_has_cid and meta_has_cid and mass_col:
    print("\nCross-matching by CID...")
    hd_cid = hd.get('CID', hd.get('SNID', hd.get('cid')))
    meta_cid = meta.get('CIDint', meta.get('CID'))
    
    mass_lookup = {}
    for i in range(N_meta):
        mass_lookup[meta_cid[i]] = meta[mass_col][i]
    
    logmass = np.array([mass_lookup.get(c, -9) for c in hd_cid])
    n_matched = np.sum(logmass > 0)
    print("  Matched %d / %d" % (n_matched, N_hd))
elif not hd_has_cid and mass_col:
    # Assume same ordering if same length
    if N_hd == N_meta:
        print("\nNo CID in HD — assuming same row ordering (N=%d)" % N_hd)
        logmass = meta[mass_col][:N_hd]
    else:
        # Try matching by zHD
        print("\nMatching by redshift (approximate)...")
        meta_z = meta.get('zHD', meta.get('zCMB'))
        if meta_z is not None:
            logmass = np.full(N_hd, -9.0)
            for i in range(N_hd):
                dz = np.abs(meta_z - hd[z_col][i])
                best = np.argmin(dz)
                if dz[best] < 0.001:
                    logmass[i] = meta[mass_col][best]
            print("  Matched %d / %d" % (np.sum(logmass > 0), N_hd))
        else:
            print("ERROR: Cannot cross-match")
            sys.exit(1)
else:
    print("ERROR: Missing CID or mass column")
    sys.exit(1)

# ── 4. Load covariance ──
print("\nLoading STAT+SYS.npz...", flush=True)
npz = np.load(os.path.join(DIST_DIR, 'STAT+SYS.npz'), allow_pickle=True)
nsn = int(npz['nsn'])
cov_flat = npz['cov']
print("  nsn = %d" % nsn)
print("  cov_flat length = %d" % len(cov_flat))
print("  Expected N*(N+1)/2 = %d" % (nsn*(nsn+1)//2))

# Reconstruct full symmetric matrix from upper triangle
if len(cov_flat) == nsn * (nsn + 1) // 2:
    print("  Upper-triangle format confirmed. Reconstructing...")
    C_full = np.zeros((nsn, nsn))
    idx = 0
    for i in range(nsn):
        for j in range(i, nsn):
            C_full[i, j] = cov_flat[idx]
            C_full[j, i] = cov_flat[idx]
            idx += 1
    print("  Full covariance: %dx%d" % C_full.shape)
elif len(cov_flat) == nsn * nsn:
    print("  Full matrix format. Reshaping...")
    C_full = cov_flat.reshape(nsn, nsn)
else:
    print("  WARNING: cov length %d doesn't match nsn=%d" % (len(cov_flat), nsn))
    print("  Trying sqrt...")
    nsn_try = int(np.sqrt(len(cov_flat)))
    if nsn_try * nsn_try == len(cov_flat):
        C_full = cov_flat.reshape(nsn_try, nsn_try)
        nsn = nsn_try
        print("  Reshaped to %dx%d" % (nsn, nsn))
    else:
        print("  Cannot determine format. Using diagonal.")
        C_full = None

# ── 5. Quality cuts ──
z = hd[z_col]
mu = hd[mu_col]
mu_err = hd[muerr_col] if muerr_col else np.full(N_hd, 0.15)

good = (logmass > 5) & (logmass < 14) & (z > 0.01) & (z < 2.0) & np.isfinite(mu)
idx_good = np.where(good)[0]
N = int(np.sum(good))

z_g = z[good]; mu_g = mu[good]; lm_g = logmass[good]; err_g = mu_err[good]
print("\n  After cuts: %d SNe" % N)

# Extract sub-covariance
if C_full is not None and nsn == N_hd:
    C = C_full[np.ix_(idx_good, idx_good)]
    has_cov = True
    print("  Sub-covariance: %dx%d" % C.shape)
    print("  Diagonal range: %.4f to %.4f" % (np.sqrt(np.min(np.diag(C))), np.sqrt(np.max(np.diag(C)))))
elif C_full is not None and nsn == N:
    C = C_full
    has_cov = True
    print("  Covariance matches cut sample: %dx%d" % C.shape)
else:
    print("  WARNING: Cov nsn=%d != N_hd=%d != N=%d. Using diagonal." % (nsn, N_hd, N))
    C = np.diag(err_g**2)
    has_cov = False

# ── 6. Split and fit ──
MASS_CUT = 10.0
low = lm_g < MASS_CUT
high = lm_g >= MASS_CUT

print("\n  Low-mass:  %d SNe, <logM*>=%.2f" % (np.sum(low), lm_g[low].mean()))
print("  High-mass: %d SNe, <logM*>=%.2f" % (np.sum(high), lm_g[high].mean()))

def mu_lcdm(z_arr, Om):
    mu_arr = np.zeros_like(z_arr)
    for i, zi in enumerate(z_arr):
        if zi < 1e-6: mu_arr[i] = -99; continue
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
        if Om < 0.01 or Om > 0.99: return 1e10
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
    
    h = 1e-4
    fp = chi2([Om_fit + h, M_fit])
    fm = chi2([Om_fit - h, M_fit])
    d2 = (fp + fm - 2*chi2_min) / h**2
    sigma_Om = 1.0/np.sqrt(max(d2, 1e-10)) if d2 > 0 else 0.1
    
    print("  %-25s: Om = %.4f +/- %.4f, chi2/dof = %.1f/%d" % 
          (label, Om_fit, sigma_Om, chi2_min, ndof))
    return Om_fit, sigma_Om, chi2_min, ndof

print("\nFitting (this takes ~10 min per subsample)...", flush=True)
print("=" * 60)

idx_low = np.where(low)[0]
idx_high = np.where(high)[0]

Om_full, sOm_full, _, _ = fit_Om(z_g, mu_g, C, "Full sample")
Om_low, sOm_low, _, _ = fit_Om(z_g[low], mu_g[low], C[np.ix_(idx_low,idx_low)], "Low-mass hosts")
Om_high, sOm_high, _, _ = fit_Om(z_g[high], mu_g[high], C[np.ix_(idx_high,idx_high)], "High-mass hosts")

dOm = Om_high - Om_low
dOm_err = np.sqrt(sOm_high**2 + sOm_low**2)
sig = abs(dOm) / dOm_err

print("\n" + "=" * 60)
print("DES-SN5YR WITH FULL COVARIANCE")
print("=" * 60)
print("  Full:      Om = %.4f +/- %.4f" % (Om_full, sOm_full))
print("  Low-mass:  Om = %.4f +/- %.4f (N=%d)" % (Om_low, sOm_low, np.sum(low)))
print("  High-mass: Om = %.4f +/- %.4f (N=%d)" % (Om_high, sOm_high, np.sum(high)))
print("  Delta_Om = %.4f +/- %.4f (%.1f sigma)" % (dOm, dOm_err, sig))
print()
print("  Previous (diagonal):  Delta_Om = -0.016 +/- 0.009 (1.9 sigma)")
print("  This run (full cov):  Delta_Om = %.3f +/- %.3f (%.1f sigma)" % (dOm, dOm_err, sig))
print("  Pantheon+:            Delta_Om = -0.051 +/- 0.015 (3.3 sigma)")

# Combined
w_p = 1/0.015**2; w_d = 1/dOm_err**2
combined = (-0.051*w_p + dOm*w_d) / (w_p + w_d)
combined_err = 1/np.sqrt(w_p + w_d)
combined_sig = abs(combined) / combined_err
print("\n  Combined (Pantheon+ + DES): Delta_Om = %.3f +/- %.3f (%.1f sigma)" % 
      (combined, combined_err, combined_sig))

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
    'uses_full_covariance': has_cov,
    'combined_with_pantheon': {
        'delta_Om': float(combined),
        'delta_Om_err': float(combined_err),
        'significance': float(combined_sig),
    }
}

outpath = os.path.join(OUTDIR, 'des_sn5yr_fullcov.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to %s" % outpath)
