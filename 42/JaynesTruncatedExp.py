# -*- coding: utf-8 -*-
"""
Created on Tue May  5 14:12:32 2020

@author: Romain
"""

#Modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv
from scipy.special import gammaincc
from scipy import optimize
from scipy.stats import expon

# Definition de l'exponentielle tronquée
def p(x, theta):
    return (x > theta) * np.exp(theta - x)

x = np.linspace(10, 18, 1000)
plt.plot(x, p(x, 10), alpha=0.6, color = "#21177D")
plt.fill(x, p(x, 10), alpha=0.6, color = "#21177D")
plt.ylim(0, 1.1)
plt.xlim(6, 18)
plt.xlabel('t')
plt.ylabel('p(t)');
#plt.savefig("C:/Users/Romain/Desktop/Mesures/Images/TruncatedExp.pdf", bbox_inches="tight")


#Approche classique
def approx_CI(D, sig=0.95):
    """Approximate truncated exponential confidence interval"""
    # use erfinv to convert percentage to number of sigma
    Nsigma = np.sqrt(2) * erfinv(sig)
    D = np.asarray(D)
    N = D.size
    theta_hat = np.mean(D) - 1
    return [theta_hat - Nsigma / np.sqrt(N),
            theta_hat + Nsigma / np.sqrt(N)]

D = [10, 12, 15]
print("approximate CI: ({0:.1f}, {1:.1f})".format(*approx_CI(D)))

#Approche classique en prennant en compte la distribution
def exact_CI(D, frac=0.95):
    """Exact truncated exponential confidence interval"""
    D = np.asarray(D)
    N = D.size
    theta_hat = np.mean(D) - 1

    def f(theta, D):
        z = theta_hat + 1 - theta
        return (z > 0) * z ** (N - 1) * np.exp(-N * z)

    def F(theta, D):
        return gammaincc(N, np.maximum(0, N * (theta_hat + 1 - theta))) - gammaincc(N, N * (theta_hat + 1))
    
    def eqns(CI, D):
        """Equations which should be equal to zero"""
        theta1, theta2 = CI
        return (F(theta2, D) - F(theta1, D) - frac,
                f(theta2, D) - f(theta1, D))
    
    guess = approx_CI(D, 0.68) # use 1-sigma interval as a guess
    result = optimize.root(eqns, guess, args=(D,))
    if not result.success:
        print("warning: CI result did not converge!")
    return result.x

np.random.seed(0)
Dlarge = 10 + np.random.random(500)
print("approx: ({0:.3f}, {1:.3f})".format(*approx_CI(Dlarge)))
print("exact: ({0:.3f}, {1:.3f})".format(*exact_CI(Dlarge)))

print("approximate CI: ({0:.1f}, {1:.1f})".format(*approx_CI(D)))
print("exact CI:       ({0:.1f}, {1:.1f})".format(*exact_CI(D)))

#Inférence
def bayes_CR(D, frac=0.95):
    """Bayesian Credibility Region"""
    D = np.asarray(D)
    N = float(D.size)
    theta2 = D.min()
    theta1 = theta2 + np.log(1. - frac) / N
    return theta1, theta2


print("common sense:         theta < {0:.1f}".format(np.min(D)))
print("frequentism (approx): 95% CI = ({0:.1f}, {1:.1f})".format(*approx_CI(D)))
print("frequentism (exact):  95% CI = ({0:.1f}, {1:.1f})".format(*exact_CI(D)))
print("Bayesian:             95% CR = ({0:.1f}, {1:.1f})".format(*bayes_CR(D)))


Nsamples = 1000
N = 3
theta = 10

np.random.seed(42)
data = expon(theta).rvs((Nsamples, N))
CIs = np.array([exact_CI(Di) for Di in data])

# find which confidence intervals contain the mean
contains_theta = (CIs[:, 0] < theta) & (theta < CIs[:, 1])
print("Fraction of Confidence Intervals containing theta: {0:.3f}".format(contains_theta.sum() * 1. / contains_theta.size))

np.random.seed(42)
N =int(1E7)
eps = 0.1

theta = 9 + 2 * np.random.random(N)
data = (theta + expon().rvs((3, N))).T
data.sort(1)
D.sort()
i_good = np.all(abs(data - D) < eps, 1)

print("Number of good samples: {0}".format(i_good.sum()))

theta_good = theta[i_good]
theta1, theta2 = bayes_CR(D)

within_CR = (theta1 < theta_good) & (theta_good < theta2)
print("Fraction of thetas in Credible Region: {0:.3f}".format(within_CR.sum() * 1. / within_CR.size))
