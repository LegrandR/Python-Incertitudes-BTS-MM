# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 09:53:02 2020

@author: Romain
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Nombre de valeurs simulees
n = 1000000


#Paramètre
alpha = 2.1e-4 #USI
DT = 4  #K
graduation = 2  #mL

#Résultat
Veq_th = 2  #mL



#tirage
LargeurTemp = alpha*DT*Veq_th
LargeurLecture = graduation

Veq = Veq_th + LargeurTemp*np.random.uniform(-100, 100, n)/100 + LargeurLecture*np.random.uniform(-100, 100, n)/100


#Stat
print('Moyenne Veq :')
print(np.mean(Veq))

print('std Veq :')
print(np.std(Veq))

print('std Veq calc :')
print(np.sqrt(LargeurLecture**2 + LargeurTemp**2)/np.sqrt(3))



#Graphique
plt.hist(Veq, 300, histtype="stepfilled",)



#mcme algorithm


def log_prior(alpha):
    if alpha[0]<0:
        return -np.inf
    return 0

def log_likelihood(alpha):
     return -0.5*((alpha[0] - Veq_th)**2)/(LargeurTemp**2+LargeurLecture**2)
 

def log_posterior(alpha):
    return log_prior(alpha) + log_likelihood(alpha)




ndim = 1  # number of parameters in the model
nwalkers = 500  # number of MCMC walkers
nburn = 2000  # "burn-in" period to let chains stabilize
nsteps = 4000  # number of MCMC steps to take

# we'll start at random locations between 0 and 2000
starting_guesses = Veq_th + (2 * np.random.rand(nwalkers, ndim) -1)/50

import emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[])
sampler.run_mcmc(starting_guesses, nsteps)

sample = sampler.chain  # shape = (nwalkers, nsteps, ndim)
sample = sampler.chain[:, nburn:, :].reshape(-1, ndim)  # discard burn-in points

# plot a histogram of the sample
fig, ax = plt.subplots()
plt.hist(sample[:,0], bins=300, histtype="stepfilled", alpha=0.3, density=True)
plt.xlim(-1, 8)
plt.xlabel("F"); plt.ylabel("Pdf(F)")
# plot a best-fit Gaussian
"""
F_fit = np.linspace(19.95, 20.05, 200)
pdf = stats.norm(np.mean(sample), np.std(sample)).pdf(F_fit)
plt.plot(F_fit, pdf, '-k')
plt.xlabel("F"); plt.ylabel("P(F)")
"""

from astroML.plotting import plot_mcmc
fig = plt.figure()
ax = plot_mcmc(sample.T, fig=fig, labels=[r'$\mu_A$', r'$\sigma_A$'], colors='k')
ax[0].plot(sample[:, 0], sample[:, 1], ',k', alpha=0.1);
