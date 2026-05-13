#!/usr/bin/env python3
"""
Paper 7: Fixed Reionisation ODE
================================
The original standalone script had a σ(M) integration bug that produced
wrong HMF normalization → τ_e ~3× too low for most models.

Fix: Use scipy.integrate.quad with fine sampling in log(k) space,
proper EH98 no-wiggle transfer function, and verify σ_8 normalization.

Output: ~/paper7_results/reion_*.npz (one per model)
Then run make_figures.py to generate fig1.
"""

import numpy as np
from scipy.integrate import quad, odeint
from scipy.interpolate import interp1d
import os, json

# ── Cosmology (Planck 2018) ──
H0 = 67.74; Om = 0.3089; Ob = 0.0486; OL = 1 - Om
sigma8 = 0.811; ns = 0.9667; h = H0 / 100
rho_m = 2.775e11 * Om * h**2  # Msun/Mpc^3

OUTDIR = os.path.expanduser('~/paper7_results')
os.makedirs(OUTDIR, exist_ok=True)

# ── EH98 no-wiggle transfer function (Eisenstein & Hu 1998 Eq. 29-31) ──
def transfer_EH98(k_hMpc):
    """k in h/Mpc. Returns T(k)."""
    # Shape parameter
    theta = 2.725 / 2.7  # CMB temp ratio
    Omh2 = Om * h**2
    Obh2 = Ob * h**2
    fb = Ob / Om
    fc = 1 - fb
    
    # Sound horizon
    s = 44.5 * np.log(9.83 / Omh2) / np.sqrt(1 + 10 * Obh2**0.75)
    
    # Alpha_gamma (Eq. 31)
    alpha_gamma = 1 - 0.328 * np.log(431 * Omh2) * fb + 0.38 * np.log(22.3 * Omh2) * fb**2
    
    # Effective shape parameter
    Gamma_eff = Om * h * (alpha_gamma + (1 - alpha_gamma) / (1 + (0.43 * k_hMpc * s)**4))
    
    q = k_hMpc * theta**2 / Gamma_eff
    
    # Transfer function (Eq. 29)
    L = np.log(2 * np.e + 1.8 * q)
    C = 14.2 + 731.0 / (1 + 62.5 * q)
    T = L / (L + C * q**2)
    return T

# ── Growth factor ──
def growth_factor(z):
    a = 1.0 / (1 + z)
    Omz = Om * (1+z)**3 / (Om*(1+z)**3 + OL)
    OLz = OL / (Om*(1+z)**3 + OL)
    return a * 2.5 * Omz / (Omz**(4./7.) - OLz + (1 + Omz/2.) * (1 + OLz/70.))

D0 = growth_factor(0)

# ── σ(M) with proper integration ──
def _sigma_sq_unnorm(R):
    """Unnormalized σ²(R) using scipy.integrate.quad in log(k) space."""
    def integrand(lnk):
        k = np.exp(lnk)
        kR = k * R
        # Top-hat window
        if kR < 1e-8:
            W = 1.0
        else:
            W = 3.0 * (np.sin(kR) - kR * np.cos(kR)) / kR**3
        T = transfer_EH98(k * h)  # k in h/Mpc for transfer function
        # P(k) ∝ k^ns * T²(k), integrand includes k³ from d³k and the Jacobian d(lnk)
        return k**3 * k**ns * T**2 * W**2 / (2 * np.pi**2)
    
    result, _ = quad(integrand, np.log(1e-5), np.log(1e4), limit=500, epsrel=1e-8)
    return result

# Build σ(M) lookup table
print("Building σ(M) lookup table...", flush=True)
_logM_arr = np.linspace(6, 16, 200)
_R_arr = (3 * 10**_logM_arr / (4 * np.pi * rho_m))**(1./3.)  # Lagrangian radius in Mpc

# First compute σ_8 for normalization
R8 = 8.0 / h  # 8 Mpc/h → Mpc
sig8_sq_raw = _sigma_sq_unnorm(R8)
norm = sigma8**2 / sig8_sq_raw
print("  σ_8 normalization factor: %.6f" % norm)
print("  σ_8 check: %.4f (should be %.4f)" % (np.sqrt(sig8_sq_raw * norm), sigma8))

# Now compute σ(M) for all masses
_sigma_arr = np.zeros(len(_logM_arr))
for i, R in enumerate(_R_arr):
    _sigma_arr[i] = np.sqrt(_sigma_sq_unnorm(R) * norm)

# Verify: σ at M ~ 10^13 should be ~1.0, at 10^8 should be ~10+
print("  σ(M=10^8) = %.2f" % np.interp(8, _logM_arr, _sigma_arr))
print("  σ(M=10^10) = %.2f" % np.interp(10, _logM_arr, _sigma_arr))
print("  σ(M=10^12) = %.2f" % np.interp(12, _logM_arr, _sigma_arr))
print("  σ(M=10^14) = %.2f" % np.interp(14, _logM_arr, _sigma_arr))

_sigma_interp = interp1d(_logM_arr, _sigma_arr, kind='cubic', fill_value='extrapolate')

def sigma_M(logM, z):
    """σ(M) at redshift z."""
    return float(_sigma_interp(logM)) * growth_factor(z) / D0

# ── Sheth-Tormen HMF ──
def dndlnM(logM, z):
    """dn/dlnM in Mpc^-3."""
    M = 10**logM
    sig = sigma_M(logM, z)
    if sig <= 0:
        return 0
    
    dc = 1.686
    nu = dc / sig
    
    # dlnσ/dlnM via finite difference
    dlogM = 0.01
    sig_p = sigma_M(logM + dlogM, z)
    sig_m = sigma_M(logM - dlogM, z)
    dlnsig_dlnM = (np.log(sig_p) - np.log(sig_m)) / (2 * dlogM)
    
    # ST mass function
    A = 0.3222; a = 0.707; p = 0.3
    f_nu = A * np.sqrt(2 * a / np.pi) * nu * (1 + (a * nu**2)**(-p)) * np.exp(-a * nu**2 / 2)
    
    return rho_m / M * f_nu * abs(dlnsig_dlnM)

# ── Abundance matching: M_UV → M_h ──
SCHECHTER = {
    5: (-20.71, 5.27e-4, -1.97), 6: (-20.52, 3.92e-4, -2.01),
    7: (-20.30, 2.78e-4, -2.05), 8: (-20.22, 1.48e-4, -2.10),
    9: (-20.35, 5.40e-5, -2.14), 10:(-20.30, 2.70e-5, -2.16),
    11:(-20.08, 1.10e-5, -2.17), 12:(-19.92, 4.80e-6, -2.19),
}

def phi_UV(MUV, z):
    zi = min(max(int(round(z)), 5), 12)
    Ms, ps, al = SCHECHTER[zi]
    x = 10**(0.4 * (Ms - MUV))
    return 0.4 * np.log(10) * ps * x**(al + 1) * np.exp(-x)

def cum_phi_UV(MUV, z):
    """Cumulative UVLF: n(< MUV) = integral from -inf to MUV."""
    M_arr = np.linspace(-30, MUV, 500)
    phi_arr = np.array([phi_UV(m, z) for m in M_arr])
    return np.trapz(phi_arr, M_arr)

def cum_HMF(logM_min, z):
    """Cumulative HMF: n(> M) = integral from logM to inf."""
    logM_arr = np.linspace(logM_min, 16, 300)
    dn_arr = np.array([dndlnM(lm, z) for lm in logM_arr])
    dlnM = (logM_arr[1] - logM_arr[0])
    return np.trapz(dn_arr, logM_arr)

def build_AM(z):
    """Build abundance matching MUV → logMh at redshift z."""
    MUV_arr = np.linspace(-25, -10, 80)
    logMh_arr = np.zeros(len(MUV_arr))
    
    for i, muv in enumerate(MUV_arr):
        n_uv = cum_phi_UV(muv, z)
        if n_uv <= 0:
            logMh_arr[i] = 15
            continue
        # Find logM such that cum_HMF(logM, z) = n_uv
        lo, hi = 8, 15
        for _ in range(50):
            mid = (lo + hi) / 2
            if cum_HMF(mid, z) > n_uv:
                lo = mid
            else:
                hi = mid
        logMh_arr[i] = (lo + hi) / 2
    
    return interp1d(MUV_arr, logMh_arr, kind='linear', fill_value='extrapolate')

# ── Build AM for z=5-12 ──
print("\nBuilding abundance matching...", flush=True)
AM = {}
for z in range(5, 13):
    AM[z] = build_AM(z)
    logMh_21 = AM[z](-21)
    logMh_18 = AM[z](-18)
    print("  z=%d: MUV=-21 → logMh=%.2f, MUV=-18 → logMh=%.2f" % (z, logMh_21, logMh_18))

def get_logMh(MUV, z):
    zi = min(max(int(round(z)), 5), 12)
    return float(np.clip(AM[zi](MUV), 8, 15))

# ── Emissivity integral ──
def LUV(MUV):
    return 10**(0.4 * (51.63 - MUV))

def emissivity(z, fesc_func, xi_func):
    """Total ionising emissivity dn_ion/dt in photons/s/Mpc^3."""
    MUV_arr = np.linspace(-25, -10, 150)
    integrand = np.zeros(len(MUV_arr))
    for i, muv in enumerate(MUV_arr):
        phi = phi_UV(muv, z)
        if phi <= 0:
            continue
        logMh = get_logMh(muv, z)
        Mh = 10**logMh
        fe = fesc_func(Mh, z)
        xi = xi_func(muv, z)
        integrand[i] = fe * xi * LUV(muv) * phi
    return np.trapz(integrand, MUV_arr)

# ── Reionisation ODE ──
nH0 = 1.88e-7  # cm^-3 (comoving hydrogen number density)
Mpc_cm = 3.0857e24
yr_s = 3.156e7

def Hz(z):
    return H0 * np.sqrt(Om * (1+z)**3 + OL)  # km/s/Mpc

def C_HII(z):
    """Clumping factor."""
    return 3.0

def reion_ode(Q, z, fesc_func, xi_func):
    """dQ/dz for ionised fraction Q."""
    if z < 4 or z > 20:
        return 0
    ndot = emissivity(z, fesc_func, xi_func)
    # Recombination rate
    alpha_B = 2.6e-13  # cm^3/s at T=10^4 K
    n_rec = C_HII(z) * alpha_B * nH0 * (1+z)**3 * Q
    # dQ/dt = ndot/nH0 - n_rec/nH0... but we want dQ/dz
    # dQ/dz = dQ/dt * dt/dz = dQ/dt * (-1)/((1+z)*H(z))
    H = Hz(z) * 1e5 / Mpc_cm  # 1/s
    dQdt = ndot / (nH0 * Mpc_cm**3) - n_rec
    dQdz = -dQdt / ((1+z) * H)
    return dQdz

def solve_reion(fesc_func, xi_func, label):
    """Solve reionisation ODE, return z, xHI, tau_e."""
    z_arr = np.linspace(20, 4, 500)
    Q = odeint(reion_ode, 0, z_arr, args=(fesc_func, xi_func), mxstep=5000)
    Q = np.clip(Q.flatten(), 0, 1)
    xHI = 1 - Q
    
    # Thomson optical depth
    sigma_T = 6.6524e-25  # cm^2
    c_cm = 2.998e10
    tau = 0
    for i in range(len(z_arr) - 1):
        z1, z2 = z_arr[i], z_arr[i+1]
        dz = abs(z2 - z1)
        zm = (z1 + z2) / 2
        Qm = (Q[i] + Q[i+1]) / 2
        ne = nH0 * (1 + zm)**3 * Qm * 1.08  # 1.08 for HeI
        H = Hz(zm) * 1e5 / Mpc_cm
        dtdz = 1 / ((1 + zm) * H)
        tau += sigma_T * ne * c_cm * dtdz * dz
    
    # Find z_mid (xHI = 0.5) and z_end (xHI = 0.01)
    z_mid = np.interp(0.5, xHI[::-1], z_arr[::-1]) if np.any(xHI < 0.5) else 0
    z_end = np.interp(0.01, xHI[::-1], z_arr[::-1]) if np.any(xHI < 0.01) else 0
    
    sig_tau = abs(tau - 0.054) / 0.007
    print("  %s: τ=%.4f (%.1fσ) z_mid=%.1f z_end=%.1f" % (label, tau, sig_tau, z_mid, z_end))
    
    return z_arr, xHI, tau

# ── Define fesc and xi models ──
xi_fid = 10**25.35  # Hz/erg

def xi_const(MUV, z):
    return xi_fid

def xi_simm24a(MUV, z):
    """Simmonds+2024a emission-line: steep z-evolution."""
    return 10**(25.02 + 0.07 * z)

def xi_simm24b(MUV, z):
    """Simmonds+2024b mass-complete: flat."""
    return 10**(0.003 * z - 0.018 * MUV + 25.98)

def fesc_const10(M, z):
    return 0.10

def fesc_const20(M, z):
    return 0.20

def fesc_profile(M, z):
    """Profile-likelihood best fit from Paper 6."""
    return np.clip(0.061 * (M / 1e10)**0.18 * ((1+z)/10)**1.98, 0, 1)

def fesc_flat(M, z):
    """αM = +0.18, αz = 0 (flat redshift evolution)."""
    return np.clip(0.061 * (M / 1e10)**0.18, 0, 1)

# ── Run all models ──
print("\nSolving reionisation ODE for each model:")
print("=" * 60)

models = [
    ('profile',       fesc_profile, xi_const,   'Steep fesc + const xi'),
    ('flat_az',        fesc_flat,    xi_const,   'Flat fesc(az=0) + const xi'),
    ('const10',        fesc_const10, xi_const,   'Const fesc=10% + const xi'),
    ('const20',        fesc_const20, xi_const,   'Const fesc=20% + const xi'),
    ('simm24a_const',  fesc_const10, xi_simm24a, 'Const fesc=10% + Simm24a xi'),
    ('simm24b_steep',  fesc_profile, xi_simm24b, 'Steep fesc + mass-complete xi'),
]

for key, fesc_f, xi_f, label in models:
    z_arr, xHI, tau = solve_reion(fesc_f, xi_f, label)
    np.savez(os.path.join(OUTDIR, 'reion_%s.npz' % key),
             z=z_arr, xHI=xHI, tau_e=tau)

# ── Also compute kernel centroids with fixed σ(M) ──
print("\nKernel centroids:")
for z in range(5, 13):
    MUV_arr = np.linspace(-25, -10, 150)
    num, den = 0, 0
    for muv in MUV_arr:
        phi = phi_UV(muv, z)
        if phi <= 0:
            continue
        logMh = get_logMh(muv, z)
        K = xi_fid * LUV(muv) * phi
        num += K * logMh
        den += K
    cent = num / den if den > 0 else 0
    print("  z=%d: <logMh>_K = %.2f" % (z, cent))

print("\nDone. Results in %s" % OUTDIR)
