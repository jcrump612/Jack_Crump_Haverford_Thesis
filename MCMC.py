"""
Author: Jack Crump
Advisor: Professor Daniel Grin

Code produced for a senior thesis in the Department of Physics and Astronomy at Haverford College

This code will produce and save MCMC chains for the fitting of an energy conserving BSBM model to Pantheon supernova data and for a LambdaCDM model also fit to that data.
"""

import numpy as np
import math
from scipy.integrate import simpson, solve_ivp
import emcee

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
z_SN_pan, mu_SN_pan =np.loadtxt('./Data/lcparam_full_long_zhel.txt',usecols=(1, 4),  unpack=True)

#sort the redshift list
z_SN_sort_pan=np.sort(z_SN_pan)
#does not include repeat values
z_SN_unique_pan=np.unique(z_SN_sort_pan)

#start other lists
mu_SN_sort_pan=[]

#run through all unique redshifts
for i in range(len(z_SN_unique_pan)):
    #redshift in question
    z=z_SN_unique_pan[i]
    #indices where this redhsift occurs
    indexes=np.where(z_SN_pan==z)[0]
    #run througn those indices
    for j in indexes:
        #find the distance modulus at that redhift
        mu=mu_SN_pan[j]
        #append to the sorted list
        mu_SN_sort_pan.append(mu)
        
mu_SN_sort_pan=np.array(mu_SN_sort_pan)

#convert the apparent magnitude to a distance modulus using an absolute magnitude from https://arxiv.org/pdf/2101.08641.pdf%C3%A2%E2%82%AC%E2%80%B9
mu_SN_sort_pan+=19.2435


#load in the inverse covariance matrix of the data also taken from https://github.com/dscolnic/Pantheon
iCovmat_pan = np.loadtxt("./Data/sys_full_long.txt")

#number of supernovae in the sample
num=int(iCovmat_pan[0])
#start a matrix
iCovmat_pan_mat=[]
#there are num columns in the matrix
for i in range(num):
    #each row in the matrix is num long
    row=np.array(iCovmat_pan[i*num+1:(i+1)*num+1])
    iCovmat_pan_mat.append(row)
iCovmat_pan_mat=np.array(iCovmat_pan_mat)


#chi squared function taking into account covariance
def chisq(Data,Theory,icovmat):
    #difference between data and theoretical model
    diff=Data-Theory
    #transpose of that difference
    diffT=diff.T
    #chi squared calculation
    chisq=diffT.dot(icovmat.dot(diff))
    #return an infinite chi squared if there is a nan
    if np.isnan(chisq)==True:
        return np.inf
    return chisq

#calculates the chi squared for the supernovae data
def SN_Chisq_Proper(params):
    #set the free parameters
    H0, Omega_m, omega, zeta=params
    #set the scalar field mass to 0
    m=0
    
    #Set the priors
    if H0<40. or H0>90. or Omega_m<0.001 or Omega_m>0.5 or omega<=0 or omega>1 or zeta<-1 or zeta>1 or np.isnan(H0)==True or np.isnan(Omega_m)==True or np.isnan(omega)==True or np.isnan(zeta)==True: 
        return np.inf
    
    #find the model distance modulus
    mu_T=distance_modulus(z_SN_sort_pan,H0,Omega_m,omega,zeta,m)
    
    #find the chi squared of this model
    SN_Chisq_Proper=chisq(mu_SN_sort_pan,mu_T,iCovmat_pan_mat)
    return SN_Chisq_Proper

#calculates the log liklihood 
def SN_loglike_covmat(params):
    #log likelihood is -0.5 of the chi squared
    SN_loglike_covmat=-0.5*SN_Chisq_Proper(params)
    #this log likelihood is multiplied by 10^4 so that the MCMC can better tell the differences between different points
    return SN_loglike_covmat*1e4


#new chi squared for LCDM model
#calculates the chi squared for the supernovae data
def SN_Chisq_Proper_LCDM(params):
    #set the free parameters
    H0, Omega_m=params
    #set the scalar field mass to 0, omega to 1, and zeta to 10^-100
    m=0
    omega=1
    zeta=1e-100
    
    #Set the priors
    if H0<40. or H0>90. or Omega_m<0.001 or Omega_m>0.5 or np.isnan(H0)==True or np.isnan(Omega_m)==True: 
        return np.inf
    
    #find the model distance modulus
    mu_T=distance_modulus(z_SN_sort_pan,H0,Omega_m,omega,zeta,m)
    
    #find the chi squared of this model
    SN_Chisq_Proper=chisq(mu_SN_sort_pan,mu_T,iCovmat_pan_mat)
    return SN_Chisq_Proper

#new log likelihood for LCDM model
#calculates the log liklihood 
def SN_loglike_covmat_LCDM(params):
    #log likelihood is -0.5 of the chi squared
    SN_loglike_covmat=-0.5*SN_Chisq_Proper_LCDM(params)
    #this log likelihood is multiplied by 10^4 so that the MCMC can better tell the differences between different points
    return SN_loglike_covmat*1e4


#set the number of parameters and walkers
ndim, nwalkers = 4, 100
#initial points chosen from a random distribution
p0 = np.random.normal([67.4,0.315,0.8,1e-100], [5,0.1,0.1,0.5],size=(nwalkers,ndim))
#number of iterations
niter=20000
#create the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, SN_loglike_covmat)
#run the MCMC
for sample in sampler.sample(p0, iterations=niter, progress=True):
    #save the chain every 100 iterations
    if sampler.iteration % 100:
        continue
    chains = sampler.get_chain()
    np.save("MCMC_chains.npy", chains)


#same procedure for the LambdaCDM chain
#set the number of parameters and walkers
ndim, nwalkers = 2, 100
#initial points chosen from a random distribution
p0 = np.random.normal([67.4,0.315], [5,0.1],size=(nwalkers,ndim))
#number of iterations
niter=5000
#create the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, SN_loglike_covmat_LCDM)
#run the MCMC
for sample in sampler.sample(p0, iterations=niter, progress=True):
    #save the chain every 100 iterations
    if sampler.iteration % 100:
        continue
    chains = sampler.get_chain()
    np.save("MCMC_LCDM_chains.npy", chains)


