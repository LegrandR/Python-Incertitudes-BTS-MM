# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 22:00:46 2020

@author: Romain
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

np.random.seed(10) #10

# Choose the "true" parameters.
a_true = 2
b_true = -5
#igma = 12

# Generate some synthetic data from the model.
N = 20
x = 0.5 + np.sort(99 * np.random.rand(N))
sigma = 2 + 20 * np.random.rand(N)
y = a_true * x + b_true
yerr = stats.norm.rvs(size = N, loc = 0, scale = abs(sigma))
y+= yerr

NbOutliers = 3
Nrand = np.random.randint(0,N-1,size = NbOutliers)
y[Nrand] = [174.5, 115.9, 95.9 ]


#tracé graphique
plt.errorbar(x, y, yerr=abs(sigma), fmt='.k', ecolor='gray', elinewidth = 1);




def log_prior(theta):
    #g_i needs to be between 0 and 1
    if (all(theta[2:] > 0) and all(theta[2:] < 1)):
        return 0
    else:
        return -np.inf  # recall log(0) = -inf

def log_likelihood(theta, x, y, e, sigma_B):
    dy = y - theta[0] - theta[1] * x
    g = np.clip(theta[2:], 0, 1)  # g<0 or g>1 leads to NaNs in logarithm
    logL1 = np.log(g) - 0.5 * np.log(2 * np.pi * e ** 2) - 0.5 * (dy / e) ** 2
    logL2 = np.log(1 - g) - 0.5 * np.log(2 * np.pi * sigma_B ** 2) - 0.5 * (dy / sigma_B) ** 2
    return np.sum(np.logaddexp(logL1, logL2))



def log_posterior(theta, x, y, e, sigma_B):
    return log_prior(theta) + log_likelihood(theta, x, y, e, sigma_B)



ndim = 2 + len(x)  # number of parameters in the model
nwalkers = 50  # number of MCMC walkers
nburn = 10000  # "burn-in" period to let chains stabilize
nsteps = 15000  # number of MCMC steps to take

# set theta near the maximum likelihood, with 
np.random.seed(0)
starting_guesses = np.zeros((nwalkers, ndim))
starting_guesses[:, :2] = np.random.normal(0, 1, (nwalkers, 2))
starting_guesses[:, 2:] = np.random.normal(0.5, 0.1, (nwalkers, ndim - 2))

import emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[x, y, sigma, 50])
sampler.run_mcmc(starting_guesses, nsteps)

sample = sampler.chain  # shape = (nwalkers, nsteps, ndim)
sample = sampler.chain[:, nburn:, :].reshape(-1, ndim)

fig = plt.subplot()
plt.plot(sample[:, 0], sample[:, 1], ',k', alpha=0.1)
plt.xlabel('intercept')
plt.ylabel('slope');


#fiting
sigma_0 = sigma * np.ones(N)
A = np.vander(x, 2)
C = np.diag(sigma_0 * sigma_0)
ATA = np.dot(A.T, A / (sigma_0 ** 2)[:, None])
cov = np.linalg.inv(ATA)
w = np.linalg.solve(ATA, np.dot(A.T, y / (sigma_0) ** 2))

a_calc=w[0]
b_calc=w[1]



fig = plt.subplot()
theta3 = np.mean(sample[:, :2], 0)
g = np.mean(sample[:, 2:], 0)
outliers = (g < 0.45)

xfit = np.linspace(0, 100)

plt.errorbar(x, y, sigma, fmt='.k', ecolor='gray')
#plt.plot(xfit, theta1[0] + theta1[1] * xfit, color='lightgray')
plt.plot(xfit, w[1] + w[0] * xfit, color='gray')
plt.plot(xfit, theta3[0] + theta3[1] * xfit, color='black')
plt.plot(x[outliers], y[outliers], 'ro', ms=20, mfc='none', mec='red')
plt.plot(x[Nrand], y[Nrand], 'ro', ms=15, mfc='none', mec='gray')
plt.title('Maximum Likelihood fit: Bayesian Marginalization');