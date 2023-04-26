# Jack_Crump_Haverford_Thesis
Contains all code used by Jack Crump for his senior thesis in the Department of Physics and Astronomy at Haverford College.

Author: Jack Crump

Advisor: Professor Daniel Grin

MCMC.py runs the MCMCs for both the BSBM model and LambdaCDM. The Type Ia supernova data needed for those MCMCs is contained within the Data folder.

getdists_plots.py can be used to generate corner plots from the MCMC chains. The chains from the MCMCs are also contained in the Data folder.

bestfit_plots.py can be used to generate the plots of the distance modulus vs redshift for the best fit parameters found in the MCMCs. This also generates a binned residual comparison between the BSBM model and the LambdaCDM model.

alpha_variation.py creates the plots that show the alpha variation at last scattering for different zeta values and the alpha variation evolution for zeta=1e-7. This also produces fractional difference plots between the energy conserved and non-conserved models for both of the prior plots.

energy.py computes the electric energy, exact magnetic energy, and approximate magnetic energy for an electron in a 3D infinite square well. It produces a plot to vary the number of points in the Monte Carlo to check for convergence and for increasing energy levels in the infnite sum to check convergence of that sum. The numpy arrays for the sum plot are contained in the Data folder as they take a long time to run.

All plots produced can also be found in the Figures folder.

Any questions can be directed to jcrump612@gmail.com
