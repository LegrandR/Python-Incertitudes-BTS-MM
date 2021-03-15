# -*- coding: utf-8 -*-
"""
Module pour le traitement de données et l'évaluation des incertitudes.

Fonctions :
    - LinearReg

@author: R. Legrand
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import emcee

from scipy.optimize import minimize


class Sol2parametres:
    def __init__(self):
        self.A = 0
        self.B = 0
        self.SigmaA = 0
        self.SigmaB = 0
        self.W = 0
        self.A_PlusSigma = 0
        self.A_MoinsSigma = 0
        self.B_PlusSigma = 0
        self.B_MoinsSigma = 0
        


        
def LinearReg(X, Y, SigmaY):
    # Linear reg
    N = len(X)
    Out = Sol2parametres()
    sigma_err = SigmaY* np.ones(N)
    A = np.vander(X, 2)
    #C = np.diag(sigma_err * sigma_err)
    ATA = np.dot(A.T, A / (sigma_err ** 2)[:, None])
    cov = np.linalg.inv(ATA)
    w = np.linalg.solve(ATA, np.dot(A.T, Y / sigma_err ** 2))
    Out.A = w[0]
    Out.B = w[1]
    Out.SigmaA = np.sqrt(cov[0, 0])
    Out.SigmaB = np.sqrt(cov[1, 1])
    Out.W = w
    return Out

def LinearRegSimple(*argsv):
    # Linear reg
    args = []
    for a in argsv:
        args.append(a);
    X = args[0]
    Y = args[1]
    SigmaY = 1;
    N = len(X)
    Out = Sol2parametres()
    sigma_err = SigmaY* np.ones(N)
    A = np.vander(X, 2)
    #C = np.diag(sigma_err * sigma_err)
    ATA = np.dot(A.T, A / (sigma_err ** 2)[:, None])
    cov = np.linalg.inv(ATA)
    w = np.linalg.solve(ATA, np.dot(A.T, Y / sigma_err ** 2))
    Out.A = w[0]
    Out.B = w[1]
    Out.SigmaA = np.sqrt(cov[0, 0])
    Out.SigmaB = np.sqrt(cov[1, 1])
    Out.W = w
    return Out


def GetNoisyData(D, S):
    Noise = abs(10*np.random.rand(len(D)))
    D += np.random.normal(0, Noise)
    return np.random.normal(D, abs(D*S))


def log_incertitudeNormale(theta, x, y, sigma):
    a, b = theta
    model = a * x + b
    sigma2 = sigma ** 2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))


def MaxVraissemblance2parametres(X, Y, SigmaY):
    SolInit = LinearReg(X, Y, SigmaY);
    Out = Sol2parametres()
    fun = lambda *args: -log_incertitudeNormale(*args)
    initial = np.array([SolInit.A, SolInit.B]) + 0.1 * np.random.randn(2)
    sln =  minimize(fun, initial, args=(X, Y, SigmaY))
    (Out.A , Out.B) = sln.x
    return Out


def MaxVraissemblance2parametresEMCEE(f, precision, *argsv):
    nwalkers = 5*precision  # number of MCMC walkers
    nburn = 500*precision  # "burn-in" period to let chains stabilize
    nsteps = nburn + 2000*precision  # number of MCMC steps to take
    SolInit = LinearRegSimple(argsv)
    pos = [SolInit.A,SolInit.B] + 1e-1 * np.random.randn(nwalkers, 2)
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, f, args=argsv)
    sampler.run_mcmc(pos, nsteps, progress=True);
    flat_samples = sampler.get_chain(discard=nburn, thin=15, flat=True)
    Out = Sol2parametres()
    mcmc = np.percentile(flat_samples[:, 0], [16, 50, 84])
    q = np.diff(mcmc)
    Out.A = mcmc[1]
    Out.A_MoinsSigma = q[0]
    Out.A_MoinsSigma = q[1]
    mcmc = np.percentile(flat_samples[:, 1], [16, 50, 84])
    q = np.diff(mcmc)
    Out.B = mcmc[1]
    Out.B_MoinsSigma = q[0]
    Out.B_MoinsSigma = q[1]    
    return Out
























"""
# Module pour le calcule des paramètres maximisant la fonction de vraisemblance
from scipy.optimize import minimize

def log_vraissemblance(theta, x, y, sigma):
    a, b = theta
    model = a * x + b
    sigma2 = sigma ** 2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

# A mettre dans la librairy python MM

nll = lambda *args: -log_vraissemblance(*args)
initial = np.array([0, 0]) + 0.1 * np.random.randn(2)
soln = minimize(nll, initial, args=(Xdata, Ydata, sigma0))
a_MV, b_MV = soln.x

print("Estimation par maximum de vraissemblance :")
print("a = {0:.3f}".format(a_MV))
print("b = {0:.3f}".format(b_MV))

"""

    
    

