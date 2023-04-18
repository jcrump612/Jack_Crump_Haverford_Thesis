"""
Author: Jack Crump
Advisor: Professor Daniel Grin

Code produced for a senior thesis in the Department of Physics and Astronomy at Haverford College

This code will compare a energy conserving and nonconserving BSBM model based on alpha variation at the surface of last scattering for a range of zeta values and compare the evolution of alpha variation over a range of redshifts.
"""

import numpy as np
import math
from scipy.integrate import simpson, solve_ivp
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

#model for the non-conserved alpha field
def model_noncon(z,y,H0,Omega_m,omega,zeta,m):
    psi = y[0]
    u = y[1]
    f=(1+z)**3
    gz=g_nopsi(f,Omega_m,z)
    #differenetial equation for psi
    psiprime = -(u/((1+z)*gz))
    #differential equation for u
    uprime = (1/(1+z)) * ((6/((omega)*gz))* Omega_m * f * math.exp(-2*psi) * zeta + 3 * u + (m**2 * psi)/gz)
    #return all results
    F_prime = [psiprime,uprime]
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

#unitless Hubble parameter for the nonconserved model
def g_nopsi(f,Omega_m,z):
    #cosmological constant in a flat universe
    Omega_l = 1-Omega_m-Omega_r
    #contribution from matter
    mass_term=Omega_m*f
    #contribution from radiation
    rad_term=(Omega_r)*(1+z)**4
    #unitless Hubble parameter
    gz=np.sqrt(mass_term+rad_term+Omega_l)
    return gz

#function for Delta alpha/alpha
def Delta_alpha_over_alpha(psi):
    Delta_alpha_over_alpha=np.exp(2*psi)-1
    return Delta_alpha_over_alpha

#function to calculate alpha variation while varying zeta
def vary_zeta_dalpha(z_max, zeta_list, H0, Omega_m, omega, m):
    dalpha_list=[]
    for zeta in zeta_list:
        arglist = (H0, Omega_m, omega, zeta, m)
        F0 = [1,0,0]
        #run the dif eq solver
        F = solve_ivp(model, [0,z_max], F0, args=arglist, dense_output=True)
        #get psi at recombination
        psi_recomb = F.sol(z_recomb)[1]
        #calculate the fractional change in alpha then
        dalpha = Delta_alpha_over_alpha(psi_recomb)
        
        dalpha_list.append(dalpha)
        
    return np.array(dalpha_list)

#function to calculate alpha variation while varying zeta for the nonconserved model
def vary_zeta_dalpha_noncon(z_max, zeta_list, H0, Omega_m, omega, m):
    dalpha_list=[]
    for zeta in zeta_list:
        arglist = (H0, Omega_m, omega, zeta, m)
        F0 = [0,0]
        #run the dif eq solver
        F = solve_ivp(model_noncon, [0,z_max], F0, args=arglist, dense_output=True)
        #get psi at recombination
        psi_recomb = F.sol(z_recomb)[0]
        #calculate the fractional change in alpha then
        dalpha = Delta_alpha_over_alpha(psi_recomb)
        
        dalpha_list.append(dalpha)
        
    return np.array(dalpha_list)

#redshift at recombination
z_recomb = 1089.80
#maximum redshift to integrate to
z_max = 1200

#maximum magnitude of zeta to test
zeta_bound = 1e-5
#number of points to use
N = 1000
#list of zetas
zeta_list = np.append(-np.flip(np.linspace(0,zeta_bound,N+1))[:-1],np.linspace(0,zeta_bound,N+1)[1:])
#Hubble constant from https://www.aanda.org/articles/aa/abs/2020/09/aa33910-18/aa33910-18.html
H0 = 67.4
#unitless matter density today from https://www.aanda.org/articles/aa/abs/2020/09/aa33910-18/aa33910-18.html
Omega_m = 0.315
#value of omega to use
omega = 1
#using a massless field
m = 0

#calculate alpha variation
dalpha = vary_zeta_dalpha(z_max, zeta_list, H0, Omega_m, omega, m)
dalpha_noncon = vary_zeta_dalpha_noncon(z_max, zeta_list, H0, Omega_m, omega, m)

#plot alpha variation
plt.plot(zeta_list,abs(dalpha),label="Conserved Model", linewidth=3)
plt.plot(zeta_list,abs(dalpha_noncon),label="Non-Conserved Model", linewidth=3)
plt.xlabel(r"$\zeta$")
plt.ylabel(r"$|\Delta\alpha/\alpha|$ at z$=1089.80$")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("variation_1089.pdf")
plt.show()

#plot the fractional difference between the two models
plt.plot(zeta_list,abs((dalpha-dalpha_noncon)/(0.5*(dalpha+dalpha_noncon))),label="Fractional Difference", linewidth=3)
plt.xlabel(r"$\zeta$")
plt.ylabel(r"Fractional Difference in $|\Delta\alpha/\alpha|$")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("variation_1089_dif.pdf")
plt.show()

#maximum redhsift to integrate to
z_max=3000
#list of redshifts
z_list=np.linspace(0,z_max,10000)
#value of zeta to test
zeta=1e-7
#parameters to test
arglist = (H0, Omega_m, omega, zeta, m)

#initial values
F0 = [1,0,0]
#solve the differential equations
F = solve_ivp(model, [0,z_max], F0, args=arglist, dense_output=True)
#solve for alpha variation
dalpha_evolution=Delta_alpha_over_alpha(F.sol(z_list)[1])
#plot for the conserved model
plt.plot(z_list,abs(dalpha_evolution),label="Conserved Model",linewidth=3)

#initial values for the nonconserved model
F0 = [0,0]
#solve the differential equations for the nonconserved model
F_noncon= solve_ivp(model_noncon, [0,z_max], F0, args=arglist, dense_output=True)
#get alpha variation for non conserved model
dalpha_noncon_evolution=Delta_alpha_over_alpha(F_noncon.sol(z_list)[0])
#plot the non conserved model
plt.plot(z_list,abs(dalpha_noncon_evolution),label="Non-Conserved Model", linewidth=3)

plt.vlines(z_recomb,0,np.nanmax([np.nanmax(abs(dalpha_evolution)),np.nanmax(abs(dalpha_noncon_evolution))]),'k',linestyle=':',linewidth=5, label="Surface of Last Scattering")
plt.xlabel(r"Redshift (z)")
plt.ylabel(r"$|\Delta\alpha/\alpha|$")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("variation_z.pdf")
plt.show()

#fractional difference between the models
frac_dif=abs((dalpha_evolution-dalpha_noncon_evolution)/(0.5*(dalpha_evolution+dalpha_noncon_evolution)))

#plot the fractional difference
plt.plot(z_list,frac_dif,label="Fractional Difference", linewidth=3)
plt.vlines(z_recomb,0,np.nanmax(frac_dif),'k',linestyle=':',linewidth=5, label="Surface of Last Scattering")
plt.xlabel(r"Redshift (z)")
plt.ylabel(r"Fractional Difference in $|\Delta\alpha/\alpha|$")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("variation_z_dif.pdf")
plt.show()
