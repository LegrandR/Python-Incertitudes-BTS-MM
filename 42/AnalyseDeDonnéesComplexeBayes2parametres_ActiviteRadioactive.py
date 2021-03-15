# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 21:48:04 2020

@author: Romain
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 15:19:58 2020

@author: Romain
"""


# Création d'une série de données pour l'accélération de la pesanteur
import numpy as np
from scipy import stats

# Pour la répétabilité:
# les données sont aléatoires mais fixées avec ce paramètre
np.random.seed(42) 

N = 100 # Nombre de mesures
# Valeur de référence de l'activité radioactive
# de l'échantillon en coups par seconde et écart type du modèle
mu_A, sigma_A = 1000, 10  

A0 = stats.norm(mu_A, sigma_A).rvs(N) #Activité de référence fluctuante
Ai = stats.poisson(A0).rvs(N)  # Réalisation des N mesures
ei = np.sqrt(Ai)  # écart type estimé pour une loi de Poisson



import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.errorbar(Ai, np.arange(N), xerr=ei, fmt='+', color = 'k', ecolor='k', alpha=0.7, elinewidth = 1.2)
#ax.vlines([A0], -2, N+2, linewidth=3, alpha=0.7, color = '#21177D')
ax.set_xlabel("Activité en coups par seconde");ax.set_ylabel("Mesure");
ax.set_ylim([-2,N+2])


      
      

      
# Definition logarithmique des termes
def log_prior(theta):
        # theta[0] représente mu_A, et theta[1] est sigma_A
        # sigma_A DOIT être positif.
    if theta[1] <= 0:
        return -np.inf
    else:
        return 0

def log_likelihood(theta, Ai, ei):
    return -0.5 * np.sum(np.log(2 * np.pi * (theta[1] ** 2 + ei ** 2))
                         + (Ai - theta[0]) ** 2 / (theta[1] ** 2 + ei ** 2))

def log_posterior(theta, Ai, ei):
    return log_prior(theta) + log_likelihood(theta, Ai, ei)



# Paramètre pour le module emcee
import emcee
ndim, nwalkers = 2, 50
nsteps, nburn = 2000, 1000

starting_guesses = np.random.rand(nwalkers, ndim)
starting_guesses[:, 0] *= 1000  # start mu between 0 and 2000
starting_guesses[:, 1] *= 10    # start sigma between 0 and 20

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[Ai, ei])
sampler.run_mcmc(starting_guesses, nsteps)

sample = sampler.chain  # shape = (nwalkers, nsteps, ndim)
sample = sampler.chain[:, nburn:, :].reshape(-1, 2)


from astroML.plotting import plot_mcmc
fig = plt.figure()
ax = plot_mcmc(sample.T, fig=fig, labels=[r'$\mu_A$', r'$\sigma_A$'], colors='k')
ax[0].plot(sample[:, 0], sample[:, 1], ',k', alpha=0.1)
ax[0].plot([mu_A], [sigma_A], 'o', color='red', ms=10);


print(" mu    = {0:.0f} +/- {1:.0f}".format(sample[:, 0].mean(), sample[:, 0].std()))
print(" sigma = {0:.0f} +/- {1:.0f}".format(sample[:, 1].mean(), sample[:, 1].std()))



'''
Traitement classique d'un modèle à deux paramètres via Boostrap
'''

# Definition de la fonction vraisemblance
def log_likelihood(theta, Ai, ei):
    return -0.5 * np.sum(np.log(2 * np.pi * (theta[1] ** 2 + ei ** 2))
                         + (Ai - theta[0]) ** 2 / (theta[1] ** 2 + ei ** 2))

# maximiser la fonction vraisemblance <-> minimiser son opposée
def neg_log_likelihood(theta, F, e):
    return -log_likelihood(theta, F, e)


#Modules
import numpy as np
from scipy import optimize     
from astroML.resample import bootstrap

theta_guess = [900, 5]


'''
theta_est = optimize.fmin(neg_log_likelihood, theta_guess, args=(Ai, ei))
print("""
      Maximum likelihood estimate for {0} data points:
          mu={theta[0]:.0f}, sigma={theta[1]:.0f}
      """.format(N, theta=theta_est))
'''    
      


def fit_SousEchantillon(SousEchantillon):
    # Calcul des paramètres de maximisation de la fonction
    # de vraisemblance pour chaque échantillonage du boostrap
    return np.array([optimize.fmin(neg_log_likelihood, theta_guess,
                                   args=(Ai, np.sqrt(Ai)), disp=0)
                     for Ai in SousEchantillon])

# Re-échantillonage via un algorithme boostrap
# Et calcul des paramètres d'optimisation pour chaque sous-échantillon
# du boostrap
EchantillonsBoostrap = bootstrap(Ai, 500, fit_SousEchantillon)   

mu_Boostrap = EchantillonsBoostrap[:, 0]
sig_Boostrap = abs(EchantillonsBoostrap[:, 1])

print(" mu    = {0:.0f} +/- {1:.0f}".format(mu_Boostrap.mean(), mu_Boostrap.std()))
print(" sigma = {0:.0f} +/- {1:.0f}".format(sig_Boostrap.mean(), sig_Boostrap.std()))