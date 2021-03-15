# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 17:04:41 2021

@author: Prof
"""



""" Modules """
#Modules standard
import numpy as np  
import matplotlib.pyplot as plt
from scipy.special import erf

tau = 2
sigma = 0.5

X = np.linspace(-2,10,1000)

Py = np.exp(-X/tau)/tau * np.exp(-(sigma**3)/2/tau**2) / 2 *(1 - erf((-X+ (sigma**2) / tau) / (np.sqrt(2)*sigma)))

plt.plot(X, Py)



""" Paramètres :"""
# Permet de varier l'initialisation du générateur pseudo-aléatoire
np.random.seed(12)          

# Paramètre du modèle afine : y = ax + b
a_vrai = 2
b_vrai = 1

# Paramètres des bruits aléatoires
sigma0 = 0.5
tau = 2


""" Génération de données à partir du modèle """
N = 20
Xdata = np.sort(10 * np.random.rand(N))
Yvrai = a_vrai * Xdata + b_vrai
Ydata = np.random.normal(Yvrai, sigma0) # Bruit gaussien
Ydata +=  np.random.exponential(tau, N)


""" Tracé de data = {Xvrai, Yvrai, sigma0} et la loi vrai """
plt.figure(0)
plt.errorbar(Xdata, Ydata, yerr=sigma0, fmt=".k", capsize=0, label = "Données bruitées")
x0 = np.linspace(0, max(Xdata)+1, 500)
plt.plot(x0, a_vrai * x0 + b_vrai, "k", alpha=0.3, lw=3, label="Loi vraie")
plt.xlim(0, max(Xdata)+1)
plt.xlabel("x")
plt.ylabel("y");
plt.legend(fontsize=14)



import emcee

def log_Vraisemblance(theta, x, y, sigma):
    a, b, tau = theta
    if tau < 0.00001:
        return -np.inf
    model = a * x + b
    Yerr = y - model;
    Py = np.exp(-Yerr/tau)/tau * np.exp(-(sigma**3)/2/tau**2) / 2 *(1 - erf((-Yerr+ (sigma**2) / tau) / (np.sqrt(2)*sigma)))
    if (Py == 0).any() :
        return -np.inf
    return np.sum(np.log(Py))


theta = (2, 1, 3)
log_Vraisemblance(theta, Xdata, Ydata, sigma0)


Precision = 2

nwalkers = 5*Precision  # Paramètre EMCEE : nombre de "marcheurs"
nburn = 2000*Precision  # Paramètre EMCEE : nombre de points avant stabilisation des chaînes
nsteps = nburn + 2000*Precision  # Paramètre EMCEE : nombre de pas

pos = [2,1,3] + 1e-1 * np.random.randn(nwalkers, 3) # Position initiale permettant d'accélérer la convergence
nwalkers, ndim = pos.shape # Paramètre EMCEE

#Algorithme EMCEE
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_Vraisemblance, args=(Xdata, Ydata, sigma0))
sampler.run_mcmc(pos, nsteps, progress=True);
flat_samples = sampler.get_chain(discard=nburn, thin=15, flat=True)    


