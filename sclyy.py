'''
sclyy.cpp simple computation of Cl_yy by Samuel Flender
this code computes:
Cl_yy = int dz dV/dz/dOmega int dM dn/dM (y_ell(M,z))^2
which is mathematically equivalent to 
Cl_yy = (1/(4*pi)) sum_ij DeltaN(M_i,z_j) y_ell(M_i,z_j)^2
where DeltaN is the binned halo count on the lightcone
'''

import numpy as np
import scipy.stats as stats
import math
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import pyfits
import matplotlib.pyplot as plt
import time
from numba import jit

#---------------
#---functions---
#---------------
@jit(nopython=True)
def compute_c200(M200, z):
    #Duffy 2008 relation, M200 in Msol/h
    conc200 = (5.71 / (1. + z)**0.47) * (M200 / 2e12)**-0.084
    return conc200

@jit(nopython=True)
def compute_battaglia_profile(x, M200, R200, z, rho_critical, Omega_b, Omega_M ):
    #Battaglia et al (2012) Eq. 10.
	#returns the pressure profile in keV/cm^3 at x=R/R200, for mass M200c in Msol, R200 in Mpc.

    alpha = 1.0
    gamma = -0.3
    P200 = 200. * rho_critical * Omega_b * Gnewt * M200 / Omega_M / 2. / R200 #Msun km^2 / Mpc^3 / s^2
    P0 = 18.1 * ((M200 / 1e14)**0.154 * (1. + z)**-0.758)
    xc = 0.497 * ((M200 / 1e14)**-0.00865 * (1. + z)**0.731)
    beta = 4.35 * ((M200 / 1e14)**0.0393 * (1. + z)**0.415)

    pth = P200 * P0 * (x / xc)**gamma * (1. + (x/xc))**(-1. * beta) #(km/s)^2 M_sun / Mpc^3
    pth *= ( Msol2kg * Joule2keV * 1e6 / Mpc2cm**3 ) #keV/cm^3
    p_e = pth * 0.518 #For Y=0.24; see Vikram, Lidz & Jain
    
    return p_e

@jit(nopython=True)
def compute_yell(ell, M200, z, D_A, rho_critical, Omega_b, Omega_M, h):
	'''
	input: 
	ell: multipole
	M200: mass in Msol
	z: redshift
	D_A: angular diameter distance in Mpc
	rho_critical: critical density at redshift z in Msol/Mpc^3
	Omega_b, Omega_M, h: cosmology parameters

	output: y_ell
	'''

	R200 = ( M200 / (4.0/3.0 * np.pi * 200.0 * rho_critical ) )**(1.0/3.0)
	c200 = compute_c200(M200*h,z)
	R_s  = R200/c200
	ells = D_A/R_s

	xarr = np.linspace( -5, 2, 10000)
	xarr = 10**xarr

	pressure_profile = compute_battaglia_profile(xarr/c200, M200, R200, z, rho_critical, Omega_b, Omega_M)
	arg = (ell*xarr/ells)
	yell=0.0
	for i in range(1,len(xarr)):
		yell += xarr[i]**2 * np.sin(arg[i])/arg[i] * (xarr[i]-xarr[i-1]) * pressure_profile[i]

	yell *= sigma_T_cm2/rest_electron_kev *  4*np.pi*(R_s*Mpc2cm) / ells**2

	return yell

@jit(nopython=True)
def compute_clyy(ell,counts,M_means,z_means,D_A,rho_crit,Omega_b,Omega_M,h):
	Clyy=0.0
	for i in range(0,len(M_means)):
		for j in range(0,len(z_means)):
			if counts[i][j]>0:
				Clyy += counts[i][j] * compute_yell(ell,M_means[i][j],z_means[i][j],D_A[i][j],rho_crit[i][j],Omega_b,Omega_M,h) **2.0 
	Clyy /= 4.0*math.pi
	return Clyy


#---------------
#---main code---
#---------------
if __name__ == '__main__':
	start_time = time.time()

	#---constants---
	Msol2kg = 1.9889e30 #kg
	Mpc2cm = 3.0856e24 #cm 
	Gnewt = 4.3e-9 #Mpc Mo^-1 (km/s)^2 
	Joule2keV = 6.24e15
	sigma_T_cm2 = 6.6524e-25 #cm^2
	rest_electron_kev = 511.0 #keV

	#---cosmo params
	Omega_CDM=0.220
	Omega_b_reduced=0.02258
	h=0.71
	Omega_b = Omega_b_reduced/h**2
	Omega_M = Omega_CDM + Omega_b

	#---I/O
	input_lightcone = "c06.fits"
	output_file = "test.dat"
	ell_array = [500,1000,2000,3000,4000,5000,6000,8000,10000]

	#---binning params
	logxmin = -5 #for binning of the radial pressure profile
	logxmax = 2
	logxbins = 10000
	nbins_z = 50 #for binning of redshift and mass
	nbins_M = 50


	#---main code starts here

	cosmo = FlatLambdaCDM(H0=100.0*h, Om0=Omega_M)
	print "reading data..."
	hdulist=pyfits.open(input_lightcone)
	x = hdulist[1].data
	M200 = x.field("M200RED") #Msol/h
	M200 /= h # now in Msol.
	z = x.field("REDSHIFT")
	comov_dist = x.field("COMOV_DIST") #Mpc/h
	comov_dist /= h # now in Mpc.

	zmin = min(z)
	zmax = max(z)
	Mmin = min(M200)
	Mmax = max(M200)

	print "Mmin, Mmax, zmin, zmax:", Mmin, Mmax, zmin, zmax

	#M_edges = np.linspace(Mmin,Mmax,nbins_M+1)
	#z_edges = np.linspace(zmin,zmax,nbins_z+1)
	M_edges = np.logspace(np.log10(Mmin),np.log10(Mmax),nbins_M+1)
	z_edges = np.logspace(np.log10(zmin),np.log10(zmax),nbins_z+1)

	print "computing histogram..."
	counts, dummy, dummy = np.histogram2d( M200, z, [M_edges,z_edges] )

	print "computing bin means..."
	means, dummy, dummy, dummy = stats.binned_statistic_2d( M200,z, [M200,z,comov_dist], statistic='mean', bins=[M_edges,z_edges])

	M_means = means[0]
	z_means = means[1]
	d_means = means[2]

	print "computing rho_crit, comov_dist, D_A..."
	rho_crit = cosmo.critical_density(z_means) /u.cm.to(u.Mpc) *u.g.to(u.Msun) /u.cm.to(u.Mpc) /u.cm.to(u.Mpc) /u.g *u.cm**3 
	D_A = d_means / (1+z_means)

	print "computing Clyy..."

	f = open(output_file, 'w')

	print "here is ell, D_l (150GHz) / muK^2:"
	for ell in ell_array:
	#	Clyy = sum([sum([counts[i][j] * compute_yell(ell,M_means[i],z_means[j],D_A[j],rho_crit[j],Omega_b,Omega_M,h) **2.0 for j in range(0,nbins_z)]) for i in range(0,nbins_M)])
		Clyy = compute_clyy(ell,counts,M_means,z_means,D_A,rho_crit,Omega_b,Omega_M,h)
		print ell, Clyy * 6.74 * 1e12 * ell * (ell+1) / (2.0*math.pi)
		f.write(str(ell)+" "+str(Clyy)+"\n")


	f.close()  

	print("--- runtime %s seconds ---" % (time.time() - start_time))