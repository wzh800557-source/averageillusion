#!/usr/bin/env python3
"""
Paper 7: Fig 1 — Reionisation histories calibrated to validated τ_e.

Instead of solving the full ODE (which is sensitive to emissivity normalization,
clumping factor, and unit conversions), we use the standard tanh parameterization
(same as Planck 2018) and calibrate z_re to reproduce each model's τ_e.

This is the standard approach in overview/theory papers:
  Q(z) = 0.5 * (1 + tanh((y_re - y) / δy))
  where y = (1+z)^1.5, δy = 1.5*sqrt(1+z_re)*δz

Reference: Planck 2018 VI, Eq. 3 (Lewis 2008 parameterization)
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ── Cosmology ──
H0 = 67.74; Om = 0.3089; Ob = 0.0486; OL = 1 - Om
h = H0 / 100; Yp = 0.245
nH0_cm3 = (1 - Yp) * Ob * 1.878e-29 * h**2 / 1.673e-24  # comoving H density (cm^-3)
sigma_T = 6.6524e-25  # cm^2
c_cm = 2.998e10  # cm/s
Mpc_cm = 3.0857e24

def Hz(z):
    return H0 * np.sqrt(Om * (1+z)**3 + OL) * 1e5 / Mpc_cm  # s^-1

# ── Tanh reionisation model (Planck parameterization) ──
def Q_tanh(z, z_re, dz=0.5):
    """Ionised fraction. dz controls transition width."""
    y = (1 + z)**1.5
    y_re = (1 + z_re)**1.5
    dy = 1.5 * np.sqrt(1 + z_re) * dz
    return 0.5 * (1 + np.tanh((y_re - y) / dy))

def tau_e_from_zre(z_re, dz=0.5):
    """Compute Thomson optical depth for a tanh model."""
    def integrand(z):
        Q = Q_tanh(z, z_re, dz)
        ne = nH0_cm3 * (1 + z)**3 * Q * 1.08  # 1.08 for singly ionised He
        return sigma_T * ne * c_cm / ((1 + z) * Hz(z))
    result, _ = quad(integrand, 0, 30, limit=200)
    return result

def find_zre_for_tau(tau_target, dz=0.5):
    """Find z_re that gives the target τ_e."""
    def residual(z_re):
        return tau_e_from_zre(z_re, dz) - tau_target
    return brentq(residual, 4, 20)

# ── Validated τ_e values from Paper 6 + cluster runs ──
models = [
    # (key, τ_e, dz, color, linestyle, label)
    ('profile',      0.047, 0.5, 'forestgreen', '-',  r'Steep $\alpha_z=2$'),
    ('const10',      0.072, 0.5, 'gray',        '--', r'Constant $f_{\rm esc}=10\%$'),
    ('simm24a',      0.058, 0.4, 'purple',      ':',  r'Simmonds+24a $\xi_{\rm ion}$ + const $f_{\rm esc}$'),
    ('steep_mc',     0.044, 0.5, 'darkorange',  '-.', r'Steep $f_{\rm esc}$ + mass-complete $\xi_{\rm ion}$'),
]

# ── Calibrate and generate curves ──
print("Calibrating tanh models to validated tau_e values:")
print("=" * 60)

results = {}
for key, tau, dz, col, ls, lbl in models:
    z_re = find_zre_for_tau(tau, dz)
    tau_check = tau_e_from_zre(z_re, dz)
    z_arr = np.linspace(4, 20, 500)
    Q_arr = np.array([Q_tanh(z, z_re, dz) for z in z_arr])
    xHI = 1 - Q_arr
    
    # Find z_mid and z_end
    z_mid = np.interp(0.5, Q_arr[::-1], z_arr[::-1])
    z_end = np.interp(0.99, Q_arr[::-1], z_arr[::-1])
    
    print("  %s: tau=%.4f -> z_re=%.2f, z_mid=%.1f, z_end=%.1f (check: tau=%.4f)" 
          % (lbl[:35], tau, z_re, z_mid, z_end, tau_check))
    
    results[key] = {'z': z_arr, 'xHI': xHI, 'z_re': z_re, 'tau': tau, 'dz': dz}

# ── Observational constraints (properly sourced) ──
# Each point: (z, x_HI, err_lo, err_hi, label)
obs = [
    # Lyα forest dark pixel fraction
    (5.6, 0.01, 0.01, 0.01, 'McGreer+15'),
    (5.9, 0.06, 0.05, 0.05, 'McGreer+15'),
    # Lyα EW distribution / damping wing
    (6.7, 0.15, 0.10, 0.15, 'Mason+18'),
    (7.0, 0.28, 0.15, 0.18, 'Hoag+19'),
    (7.5, 0.49, 0.11, 0.11, 'Mason+19'),
    # QSO damping wings
    (7.1, 0.40, 0.19, 0.21, 'Wang+20'),
    (7.5, 0.56, 0.18, 0.15, 'Greig+22'),
    # GRB / high-z constraints
    (8.0, 0.65, 0.15, 0.15, 'Greig+22'),
    (9.0, 0.88, 0.08, 0.05, 'Umeda+24'),
]

obs_z = [o[0] for o in obs]
obs_x = [o[1] for o in obs]
obs_el = [o[2] for o in obs]
obs_eh = [o[3] for o in obs]

# ── Plot ──
print("\nGenerating figure...")
fig, ax = plt.subplots(figsize=(7, 5))
plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})

# Plot observational constraints
ax.errorbar(obs_z, obs_x, yerr=[obs_el, obs_eh], fmt='s', color='royalblue',
            ms=7, capsize=4, label='Observations', zorder=10, markeredgecolor='navy')

# Plot Planck tau_e constraint band
tau_planck = 0.054
tau_planck_err = 0.007
z_re_planck = find_zre_for_tau(tau_planck)
z_re_lo = find_zre_for_tau(tau_planck - tau_planck_err)
z_re_hi = find_zre_for_tau(tau_planck + tau_planck_err)
z_plot = np.linspace(4, 16, 300)
ax.fill_betweenx([0, 1], z_re_lo, z_re_hi, alpha=0.08, color='gold', zorder=0)
ax.axvline(z_re_planck, color='goldenrod', ls=':', lw=1, alpha=0.5)
ax.text(z_re_planck + 0.15, 0.02, r'Planck $z_{\rm re}$', fontsize=8, 
        color='goldenrod', alpha=0.7)

# Plot model curves
for key, tau, dz, col, ls, lbl in models:
    r = results[key]
    mask = r['z'] < 16
    ax.plot(r['z'][mask], r['xHI'][mask], ls=ls, color=col, lw=2.2,
            label=r'%s ($\tau_e=%.3f$)' % (lbl, tau))

ax.set_xlabel('Redshift $z$', fontsize=13)
ax.set_ylabel(r'Neutral fraction $\bar{x}_{\rm HI}$', fontsize=13)
ax.set_xlim(5, 14); ax.set_ylim(-0.05, 1.05)
ax.legend(loc='upper left', fontsize=7, framealpha=0.9)

OUTDIR = os.path.expanduser('~/paper7_results/figures')
os.makedirs(OUTDIR, exist_ok=True)
outpath = os.path.join(OUTDIR, 'fig1_reion_history.pdf')
plt.savefig(outpath, dpi=300, bbox_inches='tight')
plt.close()
print("Saved: %s" % outpath)

# Also save to paper7_results as npz for make_figures.py compatibility
RESULTS = os.path.expanduser('~/paper7_results')
for key, tau, dz, col, ls, lbl in models:
    r = results[key]
    np.savez(os.path.join(RESULTS, 'reion_%s.npz' % key),
             z=r['z'], xHI=r['xHI'], tau_e=r['tau'])
print("Saved .npz files to %s" % RESULTS)
