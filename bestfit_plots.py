"""
Author: Jack Crump
Advisor: Professor Daniel Grin

Code produced for a senior thesis in the Department of Physics and Astronomy at Haverford College

This code will produce bestfit and residual plots for parameters coming out of MCMC chains produced separately in MCMC.py.
"""

import numpy as np
import math
from scipy.integrate import simpson, solve_ivp
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt

#radiation energy density from https://www.aanda.org/articles/aa/abs/2020/09/aa33910-18/aa33910-18.html
Omega_r = 5.44e-5

#model for the conserved alpha field
def model(z,y,H0,Omega_m,omega,zeta,m):
    #set the initial values
    f=y[0]
    psi=y[1]
    u=y[2]
    #calculate the unitless Hubble parameter
    gz=g(f,psi,u,Omega_m,omega,zeta,z)
    #differential equation for f
    fprime = (f/(1+z)) *((2/gz) * u * math.exp(-2*psi) * zeta + 3)
    #differenetial equation for psi
    psiprime = -(u/((1+z)*gz))
    #differential equation for u
    uprime = (1/(1+z)) * ((6/((omega)*gz))* Omega_m * f * math.exp(-2*psi) * zeta + 3 * u + (m**2 * psi)/gz)
    #return all results
    F_prime = [fprime,psiprime,uprime]
    return F_prime

#unitless Hubble parameter
def g(f,psi,u,Omega_m,omega,zeta,z):
    #cosmological constant energy density for a flat univers
    Omega_l = 1-Omega_m-Omega_r
    #contribution from matter
    mass_term=Omega_m*f*(1+abs(zeta)*np.exp(-2*psi))
    #contribution from the scalar field
    psi_term=(1/6)*(omega)*(u**2)
    #contribution from radiation
    rad_term=(Omega_r)*np.exp(-2*psi)*(1+z)**4
    #unitless Hubble parameter
    gz=np.sqrt(mass_term+psi_term+rad_term+Omega_l)
    return gz

#Hubble parameter
def Hubble(f,psi,u,H0,Omega_m,omega,zeta,z):
    H=H0*g(f,psi,u,Omega_m,omega,zeta,z)
    return H


#integrand for calculating the comoving distance
def deta(z_list,H0,Omega_m,omega,zeta,m):   
    #speed of light
    c=2.998e5 #km/s
    
    #initial conditions
    f0=1
    psi0=0
    u0=0
    F0=[f0,psi0,u0]
    
    #solve the differential equations
    F=solve_ivp(model,[z_list[0],z_list[-1]], F0, args=(H0,Omega_m,omega,zeta,m),dense_output=True)
    
    #calculate each component
    f, psi, u = F.sol(z_list)
    
    #calculate the hubbler parameter
    H_list = Hubble(f,psi,u,H0,Omega_m,omega,zeta,z_list)
    
    return c/H_list

#function to calculate the distance modulus
def distance_modulus(z_list,H0,Omega_m,omega,zeta,m):
    #bumber of points to integrate over
    N = 10000
    #expected maximum distance from actual redshift of interest given the number of points
    err = 2*(z_list[-1]-z_list[0])/N
    #redshifts to integrate over
    z_int = np.linspace(0,z_list[-1],N)
    #obtain the value of the integrand at those redshifts
    deta_list=deta(z_int,H0,Omega_m,omega,zeta,m)
    #list for distance muduli
    mu_T=[]
    #need to fint the distnace modulus at each of these redshifts
    for i in range(len(z_list)):
        #redshift value in question
        z = z_list[i]
        #place in the integrand list that best matches this redshift value
        j = np.where((z_int > z-err) & (z_int < z+err))[0][0]
        #takes the two lists only up to that point
        z_int_prime = z_int[:j+1]
        deta_int = deta_list[:j+1]
        #integrates using Simpson's rule to get comoving distance
        comoving_distance = simpson(deta_int, z_int_prime)
        #convert to luminosity distance
        luminosity_distance = (1+z) * comoving_distance
        #convert to distance modulus
        mu=5 * np.log10(luminosity_distance) + 25
        #append to final list
        mu_T.append(mu)
    mu_T=np.array(mu_T)
    return mu_T


#read data from https://github.com/dscolnic/Pantheon
z_SN_pan, mu_SN_pan, Delta_mu_SN_pan =np.loadtxt("./Data/lcparam_full_long_zhel.txt",usecols=(1, 4, 5),  unpack=True)

#sort the redshift list
z_SN_sort_pan=np.sort(z_SN_pan)
#does not include repeat values
z_SN_unique_pan=np.unique(z_SN_sort_pan)

#start other lists
mu_SN_sort_pan=[]
Delta_mu_SN_sort_pan=[]

#run through all unique redshifts
for i in range(len(z_SN_unique_pan)):
    #redshift in question
    z=z_SN_unique_pan[i]
    #indices where this redhsift occurs
    indexes=np.where(z_SN_pan==z)[0]
    #run througn those indices
    for j in indexes:
        #find the distance modulus and error at that redhift
        mu=mu_SN_pan[j]
        Delta_mu=Delta_mu_SN_pan[j]
        #append to the sorted list
        mu_SN_sort_pan.append(mu)
        Delta_mu_SN_sort_pan.append(Delta_mu)
        
mu_SN_sort_pan=np.array(mu_SN_sort_pan)
Delta_mu_SN_sort_pan=np.array(Delta_mu_SN_sort_pan)

#convert the apparent magnitude to a distance modulus using an absolute magnitude from https://arxiv.org/pdf/2101.08641.pdf%C3%A2%E2%82%AC%E2%80%B9
mu_SN_sort_pan+=19.2435

#calculate the distance modulus for the bestfit BSBM parameters
mu_T=distance_modulus(z_SN_sort_pan,69.0,0.175,8.8,0.82,0)

#plot the BSBM bestfit
plt.figure(num=None, figsize=(14, 8), dpi=80, facecolor='w', edgecolor='k')
#plot the standard cosmological model predictions
plt.scatter(z_SN_sort_pan,mu_T,c='k',zorder=5,linestyle="-",label="BSBM Parameters")
#plot the supernovae data
plt.errorbar(z_SN_sort_pan,mu_SN_sort_pan,yerr=Delta_mu_SN_sort_pan, linestyle="",
             label='Pantheon Supernovae Data',c='violet')
plt.xlabel("Redshift ($z$)")
plt.ylabel("Distance Modulus ($\mu$)")
plt.legend()
plt.savefig("bestfit_BSBM.pdf")
plt.show()

#calculate BSBM residuals with the data
BSBM_resid=mu_SN_sort_pan-mu_T


#calculate the distance modulus for bestfit LCDM parameters
mu_T=distance_modulus(z_SN_sort_pan,74.0,0.321,1,1e-100,0)

#plot the LCDM bestfit
plt.figure(num=None, figsize=(14, 8), dpi=80, facecolor='w', edgecolor='k')
#plot the standard cosmological model predictions
plt.scatter(z_SN_sort_pan,mu_T,c='k',zorder=5,linestyle="-",label="$\Lambda$CDM Parameters")
#plot the supernovae data
plt.errorbar(z_SN_sort_pan,mu_SN_sort_pan,yerr=Delta_mu_SN_sort_pan, linestyle="",
             label='Pantheon Supernovae Data',c='violet')
plt.xlabel("Redshift ($z$)")
plt.ylabel("Distance Modulus ($\mu$)")
plt.legend()
plt.savefig("bestfit_LCDM.pdf")
plt.show()

#calculate LCDM residuals
LCDM_resid=mu_SN_sort_pan-mu_T
#number of bins to bin data
bin_N=70
#binned residuals
LCDM_resid_binned, bin_edges, LCDM_binnumber = binned_statistic(z_SN_sort_pan,LCDM_resid,statistic='mean',bins=bin_N)
BSBM_resid_binned, bin_edges, BSBM_binnumber = binned_statistic(z_SN_sort_pan,BSBM_resid,statistic='mean',bins=bin_N)

#find the bin centers
bin_centers=[]
for i in range(len(bin_edges)-1):
    center=(bin_edges[i]+bin_edges[i+1])/2
    bin_centers.append(center)
bin_centers=np.array(bin_centers)

#bin the corresponding errors for the residuals
LCDM_errors_binned=[]
BSBM_errors_binned = []
for i in range(len(LCDM_resid_binned)):
    LCDM_bin_error=0
    BSBM_bin_error=0
    for j in range(len(Delta_mu_SN_sort_pan)):
        if LCDM_binnumber[j]==i+1:
            LCDM_bin_error+=(Delta_mu_SN_sort_pan[i])**2
        if BSBM_binnumber[j]==i+1:
            BSBM_bin_error+=(Delta_mu_SN_sort_pan[i])**2
    LCDM_errors_binned.append(np.sqrt(LCDM_bin_error))
    BSBM_errors_binned.append(np.sqrt(BSBM_bin_error))
LCDM_errors_binned=np.array(LCDM_errors_binned)
BSBM_errors_binned=np.array(BSBM_errors_binned)

#remove nan values from bin centers and errors
bin_centers_nonan=[]
LCDM_errors_binned_nonan=[]
BSBM_errors_binned_nonan=[]
for i in range(len(bin_centers)):
    if np.isnan(LCDM_resid_binned[i])==False:
        bin_centers_nonan.append(bin_centers[i])
        LCDM_errors_binned_nonan.append(LCDM_errors_binned[i])
        BSBM_errors_binned_nonan.append(BSBM_errors_binned[i])
bin_centers_nonan=np.array(bin_centers_nonan)
LCDM_errors_binned_nonan=np.array(LCDM_errors_binned_nonan)
BSBM_errors_binned_nonan=np.array(BSBM_errors_binned_nonan)

#remove nan values from residuals
LCDM_resid_binned = LCDM_resid_binned[~np.isnan(LCDM_resid_binned)]
BSBM_resid_binned = BSBM_resid_binned[~np.isnan(BSBM_resid_binned)]

#plot the residuals of both LCDM and BSBM
plt.figure(num=None, figsize=(14, 8), dpi=80, facecolor='w', edgecolor='k')
#plot the residual
plt.errorbar(bin_centers_nonan,LCDM_resid_binned,yerr=LCDM_errors_binned_nonan, linestyle="",
             label=r'$\Lambda$CDM Best Fit Binned Model Residuals',c='violet')
plt.plot(bin_centers_nonan,BSBM_resid_binned,'k',linewidth=3,label='BSBM Best Fit Binned Model Residuals',marker='.',ms=10)
plt.xlabel("Redshift ($z$)")
plt.ylabel("Distance Modulus ($\mu$) Residuals")
plt.legend()
plt.tight_layout()
plt.savefig("residual.pdf")
plt.show()