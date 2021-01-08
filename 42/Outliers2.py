# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:30:40 2020

@author: Romain
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

np.random.seed(10) #10

# Paramètre du model
a_true = 2
b_true = -5


# Generation de données à partir du model
N = 20
x = 0.5 + np.sort(99 * np.random.rand(N))
sigma = 2 + 20 * np.random.rand(N)
y = a_true * x + b_true
yerr = stats.norm.rvs(size = N, loc = 0, scale = abs(sigma))
y+= yerr

# Ajout de valeurs aberrantes
NbOutliers = 3
Nrand = np.random.randint(0,N-1,size = NbOutliers)
y[Nrand] = [174.5, 115.9, 95.9 ]


# tracé graphique
xmax = 20
ymax = 20
fig, ax = plt.subplots(figsize=(xmax/2.54, ymax/2.54))

ax.errorbar(x, y, yerr=abs(sigma), fmt='.k', ecolor='gray', elinewidth = 1, label = 'Données')
ax.errorbar(x[Nrand], y[Nrand], yerr=abs(sigma[Nrand]), fmt='.r', ecolor='red', elinewidth = 1, label = 'Données aberrantes')
ax.set_xlabel("x")
ax.set_ylabel("y")


# Regression linéaire
A = np.vander(x, 2)
C = np.diag(sigma * sigma)
ATA = np.dot(A.T, A / (sigma ** 2)[:, None])
cov = np.linalg.inv(ATA)
w = np.linalg.solve(ATA, np.dot(A.T, y / sigma ** 2))
print("Least-squares estimates:")
print("m = {0:.3f} ± {1:.3f}".format(w[0], np.sqrt(cov[0, 0])))
print("b = {0:.3f} ± {1:.3f}".format(w[1], np.sqrt(cov[1, 1])))


ax.plot(x, a_true * x + b_true, "b", alpha=0.4, lw=3, label="modéle initial")
ax.plot(x, np.dot(np.vander(x, 2), w), "--k", label="Régression linéaire")
ax.set_xlabel("x")
ax.set_ylabel("y")



# Probabilité inférence bayésienne
def log_prior(theta):
    #g_i needs to be between 0 and 1
    if (all(theta[2:] > 0) and all(theta[2:] < 1)):
        return 0
    else:
        return -np.inf  # recall log(0) = -inf

def log_likelihood(theta, x, y, e, sigma_B):
    dy = y - theta[0] - theta[1] * x
    g = np.clip(theta[2:], 0, 1)  # g<0 or g>1 leads to NaNs in logarithm
    for i in range(len(g)):
        if g[i]<0.5 : g[i]=0
        else : g[i]=1
    logL1 = np.log(g) - 0.5 * np.log(2 * np.pi * e ** 2) - 0.5 * (dy / e) ** 2
    logL2 = np.log(1 - g) - 0.5 * np.log(2 * np.pi * sigma_B ** 2) - 0.5 * (dy / sigma_B) ** 2
    return np.sum(np.logaddexp(logL1, logL2))

def log_posterior(theta, x, y, e, sigma_B):
    return log_prior(theta) + log_likelihood(theta, x, y, e, sigma_B)


# Note that this step will take a few minutes to run!

ndim = 2 + len(x)  # number of parameters in the model
nwalkers = 50  # number of MCMC walkers
nburn = 10000  # "burn-in" period to let chains stabilize
nsteps = 15000  # number of MCMC steps to take

# set theta near the maximum likelihood, with 
np.random.seed(0)
starting_guesses = np.zeros((nwalkers, ndim))
starting_guesses[:, 0] = np.random.normal(0, 5, (nwalkers))
starting_guesses[:, 1] = np.random.normal(0, 5, (nwalkers))
starting_guesses[:, 2:] = np.random.normal(0.5, 0.25, (nwalkers, ndim - 2))

import emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[x, y, sigma, 100])
sampler.run_mcmc(starting_guesses, nsteps, progress = True)

sample = sampler.chain  # shape = (nwalkers, nsteps, ndim)
sample = sampler.chain[:, nburn:, :].reshape(-1, ndim)


# tracé
theta3 = np.mean(sample[:, :2], 0)
g = np.mean(sample[:, 2:], 0)
outliers = (g < 0.5)

#ax.plot(x, theta3[0] + theta3[1] * x, color='black', label = "Ajustement par inférence bayésienne")
ax.plot(x[outliers], y[outliers], 'ro', ms=14, mfc='none', mec='blue', label='Données estimées comme aberrantes')
plt.title('Ajustement par regression : inférence bayésienne')





# Create some convenience routines for plotting

def compute_sigma_level(trace1, trace2, nbins=20):
    """From a set of traces, bin by number of standard deviations"""
    L, xbins, ybins = np.histogram2d(trace1, trace2, nbins)
    L[L == 0] = 1E-16
    logL = np.log(L)

    shape = L.shape
    L = L.ravel()

    # obtain the indices to sort and unsort the flattened array
    i_sort = np.argsort(L)[::-1]
    i_unsort = np.argsort(i_sort)

    L_cumsum = L[i_sort].cumsum()
    L_cumsum /= L_cumsum[-1]
    
    xbins = 0.5 * (xbins[1:] + xbins[:-1])
    ybins = 0.5 * (ybins[1:] + ybins[:-1])

    return xbins, ybins, L_cumsum[i_unsort].reshape(shape)

"""
def plot_MCMC_trace(ax, xdata, ydata, trace, scatter=False, **kwargs):
    #Plot traces and contours
    xbins, ybins, sigma = compute_sigma_level(trace[0], trace[1])
    ax.contour(xbins, ybins, sigma.T, levels=[0.683, 0.955], **kwargs)
    if scatter:
        ax.plot(trace[0], trace[1], ',k', alpha=0.1)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')
    """
    
def plot_MCMC_model(ax, xdata, ydata, trace):
    """Plot the linear model and 2sigma contours"""
    #ax.plot(xdata, ydata, 'ok')

    alpha, beta = trace[:2]
    xfit = np.linspace(-1, 100, 10)
    yfit = alpha[:, None] + beta[:, None] * xfit
    mu = yfit.mean(0)
    sig = 2 * yfit.std(0)

    ax.plot(xfit, mu, "C1", label = "Ajustement par inférence bayésienne")
    ax.fill_between(xfit, mu - sig, mu + sig, color = 'C1', alpha=0.1, label = r"Intervalle de crédibilité à $2\sigma$")#, color='lightgray')

    ax.set_xlabel('x')
    ax.set_ylabel('y')


def plot_MCMC_results(xdata, ydata, trace, colors='k'):
    """Plot both the trace and the model together"""
    plot_MCMC_model(ax, xdata, ydata, trace)

emcee_trace = sampler.chain[:, nburn:, :].reshape(-1, ndim).T
plot_MCMC_results(x, y, emcee_trace)
ax.legend(fontsize=11)
ax.set_ylim(-30, 250)

fig.savefig("C:/Users/Romain/Desktop/Mesures/Images/AjustementParInferenceBayesienne.pdf", bbox_inches='tight', transparent=True, pad_inches=0.5)


#import corner

"""
flat_samples = sampler.get_chain(discard=nburn, thin=15, flat=True)

inds = np.random.randint(len(flat_samples), size=100)
for ind in inds:
    sample = flat_samples[ind]
    ax.plot(x, sample[0] + sample[1]*x, "C1", alpha=0.1, zorder = -1)
"""