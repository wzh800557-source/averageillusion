#!/usr/bin/env python3
"""
Paper 7: THESAN Galaxy Bias Mismatch
=====================================
Reads THESAN-1 ion catalogues to compute the bias mismatch between
number-weighted and clustering-weighted averages using actual halo masses.

The key insight: UVLF-calibrated models use number-weighting (K ∝ dn/dM),
while clustering measures bias-weighting (K ∝ b(M) dn/dM).

Also implements Chakraborty & Choudhury (2025) style duty cycle:
  f_duty(M,z) = min(1, t_SF(M) / t_age(z))
  t_SF(M) ∝ (M/10^10)^0.2 × 80 Myr

Outputs: ~/paper7_results/thesan_bias.json
"""

import numpy as np
import h5py, os, json, glob

# ── Cosmology ──
H0 = 67.74; Om = 0.3089; OL = 1 - Om
sigma8 = 0.811; ns = 0.9667; h = H0/100
Mpc_cm = 3.0857e24

def Hz(z): return H0 * np.sqrt(Om*(1+z)**3 + OL)

def cosmic_age_s(z):
    """Age of universe at z, in seconds. Simple numerical integration."""
    from scipy.integrate import quad
    def integrand(zp):
        return 1.0 / ((1+zp) * Hz(zp) * 1e5 / Mpc_cm)
    t, _ = quad(integrand, z, 50, limit=200)
    return t

# ── Sheth-Tormen bias ──
# Using Eisenstein & Hu fitting form for σ(M)
def transfer_EH98(k):
    Gamma = Om*h*np.exp(-0.0486*(1+np.sqrt(2*h)/Om))
    q = k / Gamma
    L = np.log(2*np.e + 1.8*q)
    C = 14.2 + 731.0/(1+62.5*q)
    return L / (L + C*q**2)

def growth_factor(z):
    a = 1.0/(1+z)
    Omz = Om*(1+z)**3 / (Om*(1+z)**3 + OL)
    OLz = 1 - Omz
    return a*2.5*Omz / (Omz**(4./7.) - OLz + (1+Omz/2.)*(1+OLz/70.))

# Pre-compute σ(M) lookup table
_SIGMA_LOGM = np.linspace(7, 15, 500)
_SIGMA_VALS = None

def _build_sigma_table():
    global _SIGMA_VALS
    from scipy.integrate import quad
    rho_m = 2.775e11 * Om * h**2
    
    # First get σ8 normalisation
    R8 = 8.0
    def integrand(lnk, R):
        k = np.exp(lnk)
        kR = k * R
        if kR < 1e-6: W = 1.0
        else: W = 3*(np.sin(kR) - kR*np.cos(kR))/kR**3
        Tk = transfer_EH98(k)
        return k**3 * k**ns * Tk**2 * W**2 / (2*np.pi**2)
    
    sig8_sq, _ = quad(integrand, np.log(1e-4), np.log(1e3), args=(R8,), limit=200)
    norm = sigma8**2 / sig8_sq
    
    _SIGMA_VALS = np.zeros(len(_SIGMA_LOGM))
    for i, lm in enumerate(_SIGMA_LOGM):
        M = 10**lm
        R = (3*M/(4*np.pi*rho_m))**(1./3.)
        sig_sq, _ = quad(integrand, np.log(1e-4), np.log(1e3), args=(R,), limit=200)
        _SIGMA_VALS[i] = np.sqrt(sig_sq * norm)

def sigma_M(M, z):
    global _SIGMA_VALS
    if _SIGMA_VALS is None:
        _build_sigma_table()
    lm = np.log10(M)
    sig0 = np.interp(lm, _SIGMA_LOGM, _SIGMA_VALS)
    return sig0 * growth_factor(z) / growth_factor(0)

def halo_bias_ST(M, z):
    """Sheth-Tormen halo bias."""
    a, p = 0.707, 0.3
    dc = 1.686
    sig = sigma_M(M, z)
    nu = dc / sig
    return 1 + (a*nu**2 - 1)/dc + 2*p/(dc*(1 + (a*nu**2)**p))

# ── THESAN data ──
THESAN_DIR = '/nfs/mvogelsblab001/Thesan/Thesan-1/postprocessing/ion'
OUTDIR = os.path.expanduser('~/paper7_results')
os.makedirs(OUTDIR, exist_ok=True)

ion_files = sorted(glob.glob(os.path.join(THESAN_DIR, 'ion_*.hdf5')))
print(f"Found {len(ion_files)} THESAN ion snapshots")
print("Building σ(M) lookup table...", flush=True)
_build_sigma_table()
print("Done.\n")

# ── Compute bias for each snapshot ──
print(f"{'z':>6} {'N_halos':>8} {'<b>_num':>8} {'<b>_clust':>10} {'Δ(%)':>7} "
      f"{'<b>_duty':>9} {'<b>_cl_d':>9} {'Δ_duty(%)':>10} {'f_duty':>7}")
print("-" * 85)

results_by_z = {}

for fpath in ion_files:
    try:
        with h5py.File(fpath, 'r') as hf:
            z = float(hf.attrs['redshift'])
            if z < 5 or z > 13:
                continue
            
            M200 = hf['M_200'][:]
            Ndot = hf['Ndot_int'][:]  # ionising luminosity
            
            # Quality cuts
            mask = (M200 > 1e8) & np.isfinite(M200)
            M200 = M200[mask]
            Ndot = Ndot[mask]
            
            if len(M200) < 100:
                continue
            
            # Compute bias for each halo
            b_arr = np.array([halo_bias_ST(m, z) for m in M200])
            
            # -- Number-weighted (UVLF kernel) --
            b_num = np.mean(b_arr)
            
            # -- Bias-weighted (clustering kernel) --
            # <b>_clustering = <b²>/<b> = Σb²/Σb
            b_clust = np.sum(b_arr**2) / np.sum(b_arr)
            
            mismatch_no_duty = (b_clust - b_num) / b_num * 100
            
            # -- Luminosity-weighted --
            if np.sum(Ndot) > 0:
                w_lum = Ndot / np.sum(Ndot)
                b_lum = np.average(b_arr, weights=w_lum)
                b_clust_lum = np.average(b_arr**2, weights=w_lum) / np.average(b_arr, weights=w_lum)
            else:
                b_lum = b_num
                b_clust_lum = b_clust
            
            # -- With duty cycle (C&C 2025 style) --
            t_age = cosmic_age_s(z)
            t_SF_80Myr = 80e6 * 365.25 * 24 * 3600  # 80 Myr in seconds
            
            # Mass-dependent duty cycle
            f_duty_arr = np.minimum(1.0, (M200/1e10)**0.2 * t_SF_80Myr / t_age)
            
            # Number-weighted with duty cycle
            w_duty = f_duty_arr
            b_num_duty = np.average(b_arr, weights=w_duty)
            
            # Bias-weighted with duty cycle
            b_clust_duty = np.average(b_arr**2, weights=w_duty) / np.average(b_arr, weights=w_duty)
            
            mismatch_duty = (b_clust_duty - b_num_duty) / b_num_duty * 100
            
            # Luminosity × duty cycle
            w_lum_duty = Ndot * f_duty_arr
            if np.sum(w_lum_duty) > 0:
                b_num_ld = np.average(b_arr, weights=w_lum_duty)
                b_clust_ld = np.average(b_arr**2, weights=w_lum_duty) / np.average(b_arr, weights=w_lum_duty)
                mm_ld = (b_clust_ld - b_num_ld) / b_num_ld * 100
            else:
                b_num_ld, b_clust_ld, mm_ld = b_num, b_clust, mismatch_no_duty
            
            avg_duty = np.mean(f_duty_arr)
            
            print(f"{z:6.2f} {len(M200):8d} {b_num:8.3f} {b_clust:10.3f} {mismatch_no_duty:6.1f}% "
                  f"{b_num_duty:9.3f} {b_clust_duty:9.3f} {mismatch_duty:9.1f}% {avg_duty:7.3f}")
            
            # Store results binned by integer z
            zi = int(round(z))
            if zi not in results_by_z:
                results_by_z[zi] = {
                    'z_vals': [], 'n_halos': [],
                    'b_num': [], 'b_clust': [], 'mm_no_duty': [],
                    'b_num_duty': [], 'b_clust_duty': [], 'mm_duty': [],
                    'b_num_lum': [], 'b_clust_lum': [], 
                    'b_num_ld': [], 'b_clust_ld': [], 'mm_ld': [],
                    'avg_duty': []
                }
            r = results_by_z[zi]
            r['z_vals'].append(float(z))
            r['n_halos'].append(len(M200))
            r['b_num'].append(float(b_num))
            r['b_clust'].append(float(b_clust))
            r['mm_no_duty'].append(float(mismatch_no_duty))
            r['b_num_duty'].append(float(b_num_duty))
            r['b_clust_duty'].append(float(b_clust_duty))
            r['mm_duty'].append(float(mismatch_duty))
            r['b_num_lum'].append(float(b_lum))
            r['b_clust_lum'].append(float(b_clust_lum))
            r['b_num_ld'].append(float(b_num_ld))
            r['b_clust_ld'].append(float(b_clust_ld))
            r['mm_ld'].append(float(mm_ld))
            r['avg_duty'].append(float(avg_duty))
            
    except Exception as e:
        print(f"  Error on {os.path.basename(fpath)}: {e}")
        continue

# ── Summary by redshift ──
print(f"\n{'='*70}")
print("SUMMARY: Average bias mismatch by redshift")
print(f"{'='*70}")
print(f"{'z':>4} {'<b>_num':>8} {'<b>_cl':>8} {'Δ(%)':>7} {'Δ_duty(%)':>10} {'Δ_lum×duty':>11}")
print("-" * 52)

summary = {}
for zi in sorted(results_by_z.keys()):
    r = results_by_z[zi]
    avg_mm = np.mean(r['mm_no_duty'])
    avg_mm_d = np.mean(r['mm_duty'])
    avg_mm_ld = np.mean(r['mm_ld'])
    avg_bn = np.mean(r['b_num'])
    avg_bc = np.mean(r['b_clust'])
    
    print(f"{zi:4d} {avg_bn:8.3f} {avg_bc:8.3f} {avg_mm:6.1f}% {avg_mm_d:9.1f}% {avg_mm_ld:10.1f}%")
    
    summary[str(zi)] = {
        'b_num': float(avg_bn), 'b_clust': float(avg_bc),
        'mismatch_no_duty': float(avg_mm),
        'mismatch_duty': float(avg_mm_d),
        'mismatch_lum_duty': float(avg_mm_ld),
        'n_snapshots': len(r['z_vals']),
        'mean_n_halos': float(np.mean(r['n_halos'])),
    }

# ── C&C comparison ──
print(f"\nChakraborty & Choudhury (2025) report ~8% mismatch at z=6-12.")
print(f"Our THESAN measurement:")
all_mm_duty = []
for zi in sorted(results_by_z.keys()):
    if 6 <= zi <= 12:
        all_mm_duty.extend(results_by_z[zi]['mm_duty'])
if all_mm_duty:
    print(f"  Mean mismatch (with duty cycle, z=6-12): {np.mean(all_mm_duty):.1f}%")
    print(f"  Range: {np.min(all_mm_duty):.1f}% to {np.max(all_mm_duty):.1f}%")

# ── Save ──
output = {
    'summary_by_z': summary,
    'raw_by_z': {str(k): {kk: vv for kk, vv in v.items()} 
                 for k, v in results_by_z.items()},
    'method': ('Sheth-Tormen bias b(M,z) computed for each THESAN halo. '
               'Number-weighted: <b> = mean(b_i). '
               'Clustering-weighted: <b²>/<b>. '
               'Duty cycle: f_duty = min(1, (M/10^10)^0.2 * 80Myr / t_age(z)). '
               'Luminosity weights from Ndot_int.'),
}

outpath = os.path.join(OUTDIR, 'thesan_bias.json')
with open(outpath, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {outpath}")
