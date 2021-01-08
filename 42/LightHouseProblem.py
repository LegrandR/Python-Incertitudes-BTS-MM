# -*- coding: utf-8 -*-
"""
Created on Mon May  4 19:24:42 2020

@author: Romain
"""


# Data generation
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#Pour la répétabilité
np.random.seed(5)  

#Paramètre du problème
a = 5      #Position du phare le long de la côte
b = 10     #Distance du phare à la côte

c_start = -10000  #Début de la côte
c_end   =  10000  #Fin de la côte
b_max = 50        #Distance maximal du phare  

t_start = np.arctan(-(a-c_start)/b)
t_end   = np.arctan((a+c_end)/b)

#nombre de mesures
N =1000     # monter jusqu'à 1000 avec seed 5

theta = np.random.uniform(t_start, t_end, N)
c_pos = a + b * (np.tan(theta))


# Définition des probabilités pour inférences bayésiennes
def ProbPos(x, b, c_pos):
    P = 1
    for c in c_pos:
        P *= b / (3.14159 * (b ** 2 + (x - c)**2))
        P /= P.max()  #Pour normaliser
    return P
   
     
x=np.linspace(c_start,c_end,10000)
dpf = ProbPos(x, b, c_pos)


fig, ax = plt.subplots()
plt.plot(x, dpf)
plt.xlim(min(c_pos)-10, 1.2*max(c_pos)+10)
ax.vlines(c_pos, 0, 0.2, linewidth=1, alpha=0.5)
ax.vlines(c_pos.mean(), 0, 0.5, linewidth=1, alpha=0.2, colors='red')

print("La valeur la plus probable de la position du phare est {:.2f}".format(x[np.argmax(dpf)]))
print("La valeur moyenne de la position des données est {:.2f}".format(c_pos.mean()))

#algorythme emce : détermination du paramètre a
def log_prior(alpha):
    if (alpha[0] < c_start or alpha[0] > c_end):
        return -np.inf
    else:
        return 0


def log_likelihood(alpha, c_pos):
    return - np.sum(np.log(b**2 + (c_pos - alpha[0])**2))

def log_posterior(alpha, c_pos):
    return log_prior(alpha) + log_likelihood(alpha, c_pos)


ndim = 1  # number of parameters in the model
nwalkers = 200  # number of MCMC walkers
nburn = 1000  # "burn-in" period to let chains stabilize
nsteps = 2000  # number of MCMC steps to take

# we'll start at random locations between 0 and 2000
starting_guesses = (c_end - c_start) * np.random.rand(nwalkers, ndim) + c_start

import emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[c_pos])
sampler.run_mcmc(starting_guesses, nsteps)

sample = sampler.chain  # shape = (nwalkers, nsteps, ndim)
sample = sampler.chain[:, nburn:, :].ravel()  # discard burn-in points

# plot a histogram of the sample
fig, ax = plt.subplots()
plt.hist(sample, bins=300, histtype="stepfilled", alpha=0.3, density=True)
plt.xlim(-2, 15)
ax.vlines(c_pos, 0, 0.03, linewidth=1, alpha=0.5)
ax.vlines(c_pos.mean(), 0, 0.2, linewidth=1, alpha=0.2, colors='red')
ax.vlines(a, 0, 0.2, linewidth=1, alpha=1, colors='blue')
plt.xlabel("F"); plt.ylabel("Pdf(F)")
# plot a best-fit Gaussian
F_fit = np.linspace(-10, 25, 200)
pdf = stats.norm(np.mean(sample), np.std(sample)).pdf(F_fit)
plt.plot(F_fit, pdf, '-k')
plt.xlabel("F"); plt.ylabel("P(F)")






# 2 dimensions
def ProbPos(x, y, c_pos):
    P = 1
    for c in c_pos:
        P *= y / (3.14159 * (y ** 2 + (x - c)**2))
        P /= (P.max())  #Pour normaliser
    return P


x_line = np.linspace(-20,20,100)
y_line = np.linspace(0.1,b_max,100)

X_mesh, Y_mesh = np.meshgrid(x_line, y_line)



dpf = ProbPos(X_mesh, Y_mesh, c_pos)

plt.pcolor(X_mesh, Y_mesh, dpf)

result = np.where(dpf == np.amax(dpf))

print("Valeur de a la plus probable : {:.1f}.\nValeur vrai de a : {}".format(x_line[result[1]][0], a) )
print("Valeur de b la plus probable : {:.1f}.\nValeur vrai de b : {}".format(y_line[result[0]][0], b) )





#algorithme emce : détermination des paramètre a et b
ndim = 2  # number of parameters in the model
nwalkers = 200  # number of MCMC walkers
nburn = 1000  # "burn-in" period to let chains stabilize
nsteps = 3000  # number of MCMC steps to take

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


# we'll start at random locations between 0 and 2000 and c_start and c_end /alpha[1]
starting_guesses = np.random.rand(nwalkers, ndim)    

starting_guesses[:,0] = 40 *starting_guesses[:,0] -20
starting_guesses[:,1] *= 20
starting_guesses[:,1] += 5

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[c_pos])
sampler.run_mcmc(starting_guesses, nsteps)

sample = sampler.chain  # shape = (nwalkers, nsteps, ndim)
sample = sampler.chain[:, nburn:, :].reshape(-1, 2)

#tracé
from astroML.plotting import plot_mcmc
fig = plt.figure()
ax = plot_mcmc(sample.T, fig=fig, labels=[r'position sur la côte', r'distance à la côte'], colors='red')
plt.ylim(-2, 50)
plt.xlim(-20, 20)
ax[0].plot(sample[:, 0], sample[:, 1], ',k', alpha=0.1)
ax[0].plot(a, b, 'o', color='red', ms=10);