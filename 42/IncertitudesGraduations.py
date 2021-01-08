# -*- coding: utf-8 -*-
"""
Created on Sun May 10 09:00:43 2020

@author: Romain
"""


import numpy as np
from scipy import stats
import matplotlib.pyplot as plt



np.random.seed(1)  # for repeatability


N=500000
P = 10
L = N*P

x0 = stats.uniform.rvs(0,1,L)

xi = np.reshape(x0, [N,P] )


X = np.sum(xi, 1)

plt.hist(X, bins=200, histtype="stepfilled", alpha=0.3, density=True)



#Inference

N=100
pos = []
for i in range(N):
    pos.append(np.random.choice([-1,0,1]))


#mcme algorithm


def log_prior(alpha):
    return 0

def log_likelihood(alpha, pos):
     if (alpha[0] < -1 or alpha[0] > 1) :
        return -np.inf
     return 0

def log_posterior(alpha, pos):
    return log_prior(alpha) + log_likelihood(alpha, pos)




ndim = 1  # number of parameters in the model
nwalkers = 200  # number of MCMC walkers
nburn = 1000  # "burn-in" period to let chains stabilize
nsteps = 2000  # number of MCMC steps to take

# we'll start at random locations between 0 and 2000
starting_guesses = 2 * np.random.rand(nwalkers, ndim) -1

import emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[pos])
sampler.run_mcmc(starting_guesses, nsteps)

sample = sampler.chain  # shape = (nwalkers, nsteps, ndim)
sample = sampler.chain[:, nburn:, :].ravel()  # discard burn-in points

# plot a histogram of the sample
fig, ax = plt.subplots()
plt.hist(sample, bins=300, histtype="stepfilled", alpha=0.3, density=True)
plt.xlim(-2, 15)
plt.xlabel("F"); plt.ylabel("Pdf(F)")
# plot a best-fit Gaussian
F_fit = np.linspace(-10, 25, 200)
pdf = stats.norm(np.mean(sample), np.std(sample)).pdf(F_fit)
plt.plot(F_fit, pdf, '-k')
plt.xlabel("F"); plt.ylabel("P(F)")

