#!/usr/bin/env python3
"""Controlled toy example: fixed theta(M) acquires apparent evolution from kernel evolution alone."""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
logM = np.linspace(8, 14, 300)
theta = np.clip(0.3 * (10**logM / 1e10)**(-0.35), 0, 1)

ax = axes[0]
K_hi = np.exp(-0.5*((logM - 10.0)/0.7)**2) / 2.5
K_lo = np.exp(-0.5*((logM - 12.0)/0.8)**2) / 2.5
ax.plot(logM, theta, 'k-', lw=2.5, label=r'$\theta(M)$ [fixed]')
ax.fill_between(logM, 0, K_hi, alpha=0.3, color='#e74c3c', label=r'$K(M,z_{\rm high})$')
ax.fill_between(logM, 0, K_lo, alpha=0.3, color='#3498db', label=r'$K(M,z_{\rm low})$')
avg_hi = np.average(theta, weights=np.exp(-0.5*((logM-10)/0.7)**2))
avg_lo = np.average(theta, weights=np.exp(-0.5*((logM-12)/0.8)**2))
ax.axhline(avg_hi, color='#e74c3c', ls='--', lw=1.5, alpha=0.7)
ax.axhline(avg_lo, color='#3498db', ls='--', lw=1.5, alpha=0.7)
ax.set_xlabel(r'$\log(M/M_\odot)$'); ax.set_ylabel(r'$\theta(M)$ or $K(M)$')
ax.set_xlim(8, 14); ax.set_ylim(0, 0.45); ax.legend(fontsize=8)
ax.set_title('(a) Fixed parameter, evolving kernel', fontweight='bold')

ax = axes[1]
z_arr = np.linspace(5, 12, 50)
ax.plot(z_arr, 12.5 - 0.08*z_arr, 'o-', color='navy', ms=4, lw=2)
ax.set_xlabel('Redshift $z$'); ax.set_ylabel(r'Kernel centroid $\langle\log M\rangle_K$')
ax.set_title('(b) Kernel centroid shift', fontweight='bold')

ax = axes[2]
avg_theta = [np.average(theta, weights=np.exp(-0.5*((logM-(12.5-0.08*zi))/0.7)**2)) for zi in z_arr]
slope = np.polyfit(np.log((1+z_arr)/(1+z_arr[0])), np.log(np.array(avg_theta)/avg_theta[0]), 1)[0]
ax.plot(z_arr, avg_theta, 's-', color='#8e44ad', ms=5, lw=2, label=r'$\langle\theta\rangle(z)$')
ax.text(7, avg_theta[25]+0.02, r'$\alpha_z^{\rm apparent} = %.2f$' % slope, fontsize=12, color='#8e44ad', fontweight='bold')
ax.set_xlabel('Redshift $z$'); ax.set_ylabel(r'$\langle\theta\rangle(z)$'); ax.legend(fontsize=8)
ax.set_title('(c) Apparent evolution from kernel alone', fontweight='bold')

plt.tight_layout()
plt.savefig('figS_toy.pdf', dpi=300, bbox_inches='tight')
print("Saved figS_toy.pdf")
