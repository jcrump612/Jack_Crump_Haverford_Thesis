"""
Author: Jack Crump
Advisor: Professor Daniel Grin

Code produced for a senior thesis in the Department of Physics and Astronomy at Haverford College

This code will create getdist plots from the MCMC chains produced in MCMC.py.
"""

import getdist
from getdist import plots, MCSamples
import numpy as np
import matplotlib.pyplot as plt

#load in BSBM chains
chains_BSBM=np.load("./Data/MCMC_chains.npy")
#remove high outliers of omega to make the pots easier to see
chains_BSBM = chains_BSBM[chains_BSBM[:,:,2]<20]
#load in LCDM chains
chains_LCDM=np.load("./Data/MCMC_LCDM_chains.npy")

#names and labels for BSBM parametrs
names_BSBM = ["H0", "Omega_m", "omega", "zeta"]
labels_BSBM = ["H_0", "\Omega_{m,0}", "\omega", "\zeta"]

#create the BSBM samples
samples_BSBM = MCSamples(samples=chains_BSBM, names=names_BSBM, labels=labels_BSBM, ranges={"omega":(0,20)}, settings={'smooth_scale_2D':0.3})

#plot the BSBM chains
g = plots.get_subplot_plotter()
g.triangle_plot(samples_BSBM, ["H0", "Omega_m", "omega", "zeta"], filled=True, title_limit=1)
plt.tight_layout()
plt.savefig("corner_BSBM.pdf")
plt.show()

#names and labels for the LCDM parameters
names_LCDM = ["H0", "Omega_m"]
labels_LCDM = ["H_0", "\Omega_{m,0}"]

#create the LCDM samples
samples_LCDM = MCSamples(samples=chains_LCDM, names=names_LCDM, labels=labels_LCDM, settings={'smooth_scale_2D':0.3})

#plot the LCDM chains
g = plots.get_subplot_plotter()
g.triangle_plot(samples_LCDM, filled=True, title_limit=1)
plt.tight_layout()
plt.savefig("corner_LCDM.pdf")
plt.show()