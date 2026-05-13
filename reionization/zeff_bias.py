#!/usr/bin/env python3
"""Compute the <z>_K bias for each UVLF redshift bin. Result: 0.7% correction."""
import numpy as np

SCHECHTER = {5:(-20.71,5.27e-4,-1.97),6:(-20.52,3.92e-4,-2.01),7:(-20.30,2.78e-4,-2.05),
    8:(-20.22,1.48e-4,-2.10),9:(-20.35,5.40e-5,-2.14),10:(-20.30,2.70e-5,-2.16),
    11:(-20.08,1.10e-5,-2.17),12:(-19.92,4.80e-6,-2.19)}

def interp_sch(z):
    z=max(5,min(12,z)); zl=int(np.floor(z)); zh=min(zl+1,12); f=z-zl
    Ml,pl,al=SCHECHTER[zl]; Mh,ph,ah=SCHECHTER[zh]
    return (Ml*(1-f)+Mh*f, pl*(1-f)+ph*f, al*(1-f)+ah*f)

def phi(M,z):
    Ms,ps,al=interp_sch(z); x=10**(0.4*(Ms-M))
    return 0.4*np.log(10)*ps*x**(al+1)*np.exp(-x)

def K(M,z):
    p=phi(M,z)
    if p<=0: return 0
    L=10**(0.4*(51.63-M)); fe=min(1,0.061*(10**(11.9-0.38*(M+21))/1e10)**0.18*((1+z)/10)**1.98)
    return fe*10**25.35*L*p

print("z_nom  <z>_K   offset")
for zn in range(5,13):
    M_arr=np.linspace(-25,-10,100); z_arr=np.linspace(zn-0.5,zn+0.5,50)
    szK=sK=0
    for z in z_arr:
        for M in M_arr: k=K(M,z); szK+=z*k; sK+=k
    ze=szK/sK if sK>0 else zn
    print("  %d    %.3f   %+.4f" % (zn,ze,ze-zn))
