"""
Author: Jack Crump
Advisor: Professor Daniel Grin

Code produced for a senior thesis in the Department of Physics and Astronomy at Haverford College

This code will calculate the electric energy, exact magnetic energy, and approximate magnetic energy for an electron in a 3D infinite square well. It will determine the number of points needed in the Monte Carlo integration and the number of energy levels to include in the infinite sum.
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

#define some fundamental constants
eps_not = 8.854e-12 #A^2s^4/kgm^3
mu_not = 4*np.pi*(10**(-7)) #N/A^2
e = 1.602e-19 #C
h = 6.626e-34 #kgm^2/s
hbar = h/(2*np.pi) #kgm^2/s
c = 2.998e8 #m/s
m_e = 9.109e-31 #kg

#compton wavelength of an electron
eps = h/(m_e*c)

#energy eigenvalue for a particle in a 3D infinite square well
def E_n(a, nx, ny, nz):
    E = ((nx**2)+(ny**2)+(nz**2)) * ((np.pi**2)*(hbar**2))/(2*m_e*(a**2))
    return E

#energy eigenstate for a particle in a 3D infinite square well
def psi_n(x, y, z, a, nx, ny, nz):
    prefac = (2/a)**(3/2)
    x_part = np.sin(((nx*np.pi)/a)*x)
    y_part = np.sin(((ny*np.pi)/a)*y)
    z_part = np.sin(((nz*np.pi)/a)*z)
    psi = prefac * x_part * y_part * z_part
    return psi

#gradient of the energy eigenstate for a particle in a 3D infinite square well
def grad_psi_n(x, y, z, a, nx, ny, nz):
    prefac = ((2/a)**(3/2)) * (np.pi/a)
    x_term = prefac * nx * np.cos(((nx*np.pi)/a)*x) * np.sin(((ny*np.pi)/a)*y) * np.sin(((nz*np.pi)/a)*z)
    y_term = prefac * ny * np.sin(((nx*np.pi)/a)*x) * np.cos(((ny*np.pi)/a)*y) * np.sin(((nz*np.pi)/a)*z)
    z_term = prefac * nz * np.sin(((nx*np.pi)/a)*x) * np.sin(((ny*np.pi)/a)*y) * np.cos(((nz*np.pi)/a)*z)
    return [x_term, y_term, z_term]

#generates a list of [n_x,n_y,n_z] where no unique triplet is repeated (order does not matter)
def get_n_list(n_max):
    n_list=[]
    for i in range(1,n_max+1):
        for j in range(1,n_max+1):
            for k in range(1,n_max+1):
                if [i,j,k] not in n_list and [i,k,j] not in n_list and [j,i,k] not in n_list and [j,k,i] not in n_list and [k,i,j] not in n_list and [k,j,i] not in n_list:
                    n_list.append([i,j,k])
    return n_list

#integrand for the elecric energy
def EE_integrand(x, y, z, xp, yp, zp, a, nx, ny, nz):
    coefficient = (e**2)/(8*np.pi*eps_not)
    psi0 = psi_n(x, y, z, a, 1, 1, 1)
    psin = psi_n(x, y, z, a, nx, ny, nz)
    psi0prime = psi_n(xp, yp, zp, a, 1, 1, 1)
    psinprime = psi_n(xp, yp, zp, a, nx, ny, nz)
    distance = np.sqrt((x-xp)**2 + (y-yp)**2 + (z-zp)**2)
    integrand_val = (psi0*psin*psi0prime*psinprime)/distance
    return coefficient*integrand_val

#integrand for the exact magnetic energy
def EM_exact_integrand(x, y, z, xp, yp, zp, a, nx, ny, nz):
    coefficient = -((hbar**2)*(e**2)*mu_not)/(8*np.pi*(m_e**2))
    dpsin = grad_psi_n(x, y, z, a, nx, ny, nz)
    dpsi0 = grad_psi_n(xp, yp, zp, a, 1, 1, 1)
    psi0 = psi_n(x, y, z, a, 1, 1, 1)
    psin = psi_n(xp, yp, zp, a, nx, ny, nz)
    distance = np.sqrt((x-xp)**2 + (y-yp)**2 + (z-zp)**2)
    integrand_val = ((dpsin[0]*dpsi0[0] + dpsin[1]*dpsi0[1] + dpsin[2]*dpsi0[2])*psi0*psin)/distance
    return coefficient*integrand_val

#integrand for the approximate magnetic energy
def EM_approx_integrand(x, y, z, xp, yp, zp, a, nx, ny, nz):
    coefficient = ((e**2)*mu_not)/(8*np.pi*(hbar**2))
    E0 = E_n(a, 1, 1, 1)
    En = E_n(a, nx, ny, nz)
    psi0 = psi_n(x, y, z, a, 1, 1, 1)
    psin = psi_n(x, y, z, a, nx, ny, nz)
    psi0prime = psi_n(xp, yp, zp, a, 1, 1, 1)
    psinprime = psi_n(xp, yp, zp, a, nx, ny, nz)
    distance = np.sqrt((x-xp)**2 + (y-yp)**2 + (z-zp)**2)
    integrand_val = ((x*xp+y*yp+z*zp)*psi0*psin*psi0prime*psinprime)/distance
    return coefficient*((E0-En)**2)*integrand_val

#Monte Carlo method for calculating any of these integrals
def MC(func, N, a, n_max):
    V=a**6
    energy = 0
    n_list = get_n_list(n_max)
    for n in n_list:
        nx, ny, nz = n
        integral = 0
        for i in range(int(N)):
            x, y, z, xp, yp, zp = np.random.uniform(0,a,6)
            integrand = func(x, y, z, xp, yp, zp, a, nx, ny, nz)
            integral += integrand

        integral *= V/N
        if nx != ny and nx != nz and ny != nz:
            integral *= 6
        elif nx==ny==nz:
            integral *= 1
        else:
            integral *= 3
        energy += integral
    return energy

#list for number of points in the integration that I will test
N_list = np.logspace(1,6,21)
#box size
a=10*eps
#max number of energy levels to go to
n_max=2

EE_list_N = []
EM_exact_list_N = []
EM_approx_list_N =[]

#test each of these N for number of points in the integration
for N in N_list:
    EE = MC(EE_integrand, N, a, n_max)
    EM_exact = MC(EM_exact_integrand, N, a, n_max)
    EM_approx = MC(EM_approx_integrand, N, a, n_max)
    EE_list_N.append(EE)
    EM_exact_list_N.append(EM_exact)
    EM_approx_list_N.append(EM_approx)

#plot the energy vs number of points
plt.plot(N_list,EE_list_N, label="Electric Energy")
plt.plot(N_list,EM_exact_list_N, label="Exact Magnetic Energy")
plt.plot(N_list,EM_approx_list_N, label="Approximate Magnetic Energy")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Number of Points Sampled")
plt.ylabel("Energy (J)")
plt.legend()
plt.tight_layout()
plt.savefig("energy_N_convergence.pdf")


#number of points for the Monte Carlo integration
N=1e6
#box size
a=10*eps
#list for the max energy level to test
n_max_list=range(2,21)

EE_list_n = []
EM_exact_list_n = []
EM_approx_list_n =[]

#test different max energy levels
for n_max in n_max_list:
    EE = MC(EE_integrand, N, a, n_max)
    EM_exact = MC(EM_exact_integrand, N, a, n_max)
    EM_approx = MC(EM_approx_integrand, N, a, n_max)
    EE_list_n.append(EE)
    EM_exact_list_n.append(EM_exact)
    EM_approx_list_n.append(EM_approx)
    np.save("electric_E.npy",EE_list_n)
    np.save("exact_magnetic_E.npy",EM_exact_list_n)
    np.save("approx_magnetic_E.npy",EM_approx_list_n)

EE_list_n=np.load("electric_E.npy")
EM_exact_list_n=np.load("exact_magnetic_E.npy")
EM_approx_list_n=np.load("approx_magnetic_E.npy")

#plot the energy vs the maximum number of energy levels tested
plt.plot(range(2,len(EE_list_n)+2),EE_list_n, label="Electric Energy")
plt.plot(range(2,len(EM_exact_list_n)+2),EM_exact_list_n, label="Exact Magnetic Energy")
plt.plot(range(2,len(EM_approx_list_n)+2),EM_approx_list_n, label="Approximate Magnetic Energy")
plt.yscale("log")
plt.xlabel("Number of Energy Levels Used")
plt.ylabel("Energy (J)")
plt.legend()
plt.tight_layout()
plt.savefig("energy_n_convergence.pdf")

#print the final energy at the last tested energy level
print("The Electric Energy is", EE_list_n[-1], "J")
print("The Exact Magnetic Energy is", EM_exact_list_n[-1], "J")
print("The Approximate Magnetic Energy is", EM_approx_list_n[-1], "J")