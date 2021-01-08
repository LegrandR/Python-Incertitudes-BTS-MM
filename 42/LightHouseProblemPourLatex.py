# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:49:59 2020

@author: Romain
"""


# -*- coding: utf-8 -*-
"""
Created on Mon May  4 19:24:42 2020

@author: Romain
"""


# Data generation
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import emcee

#Pour la répétabilité
np.random.seed(5)  



#nombre de mesures
N = 10
P =1000     #number of measurements : monter jusqu'à 1000 avec seed 5


#Paramètre du problème
a = 5      #Position du phare le long de la côte
b = 4     #Distance du phare à la côte

c_start = -10000  #Début de la côte
c_end   =  10000  #Fin de la côte
b_max = 50        #Distance maximal du phare  

#Angle aléatoire d'émission du phare
t_start = np.arctan(-(a-c_start)/b)
t_end   = np.arctan((a+c_end)/b)
theta = np.random.uniform(t_start, t_end, P)

#Position aléatoire des détections de flash sur la côte
c_pos_full = a + b * (np.tan(theta))  
c_pos_full[2] -= 5
c_pos_full[4] = -120
c_pos_full[10] = -200
c_pos_full[80] = 5000

c_pos = c_pos_full[0:N]

# Définition des probabilités pour inférences bayésiennes
def log_prior(alpha):
    # sigma needs to be positive.
    if (alpha[1] < 1 or alpha[1]>b_max or alpha[0] < c_start or alpha[0] > c_end) :
        return -np.inf
    else:
        return 0

def log_likelihood(alpha, c_pos):
    if (alpha[1] < 1 or alpha[1]>b_max or alpha[0] < c_start or alpha[0] > c_end) :
        return -np.inf
    else:
        return - np.sum(np.log((alpha[1]**2 + (c_pos - alpha[0])**2)/alpha[1]))

def log_posterior(alpha, c_pos):
    return log_prior(alpha) + log_likelihood(alpha, c_pos)


#algorithme emce : détermination des paramètre a et b
ndim = 2  # number of parameters in the model
nwalkers = 50  # number of MCMC walkers
nburn = 3000  # "burn-in" period to let chains stabilize
nsteps = 4000  # number of MCMC steps to take

# Estimation pour l'initialisation des "walkers"
starting_guesses = np.random.rand(nwalkers, ndim)    
starting_guesses[:,0] = 50 *starting_guesses[:,0] -25
starting_guesses[:,1] *= 25
starting_guesses[:,1] += 0

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[c_pos])
sampler.run_mcmc(starting_guesses, nsteps)

sample = sampler.chain 
sample = sampler.chain[:, nburn:, :].reshape(-1, 2)

# Tracé graphique
from astroML.plotting import plot_mcmc
fig, ax = plt.subplots(figsize=(15, 8))
#plot_mcmc(sample.T, fig=fig, labels=[r'position a sur la côte', r'distance b à la côte'], colors='k')
ax.plot(sample[:, 0], sample[:, 1], ',k', alpha=0.4)
ax.set_ylim(-2, 50)
ax.set_xlim(-50,50)
ax.plot(a, b, 'o', color='red', ms=10);
ax.set_xlabel(r'position a sur la côte')
ax.set_ylabel(r'distance b à la côte')

fig.savefig("C:/Users/Romain/Desktop/Mesures/Images/ExempleEMCE.pdf", bbox_inches='tight', transparent=True, pad_inches=0.2)