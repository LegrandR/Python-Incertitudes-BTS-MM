# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 15:19:58 2020

@author: Romain
"""


# Création d'une série de données pour l'accélération de la pesanteur
import numpy as np
from scipy import stats
np.random.seed(123)  # Pour la répétabilité, les données sont aléatoires mais fixées avec ce paramètre

A0 = 1000  # Valeur de référence de l'activité radioactive de l'échantillon en coups par seconde
N = 50 # Nombre de mesures
Ai = stats.poisson(A0).rvs(N)  # Réalisation des N mesures
ei = np.sqrt(Ai)  # écart type estimé pour une loi de Poisson


import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.errorbar(Ai, np.arange(N), xerr=ei, fmt='+', color = 'k', ecolor='k', alpha=0.7, elinewidth = 1.2)
ax.vlines([A0], -2, N+2, linewidth=3, alpha=0.7, color = '#21177D')
ax.set_xlabel("Activité en coups par seconde");ax.set_ylabel("Mesure");
ax.set_ylim([-2,N+2])

plt.savefig("C:/Users/Romain/Desktop/Mesures/Images/ActiviteRadioactiveDonnees.pdf", bbox_inches='tight', transparent=True, pad_inches=0)


w = 1. / ei ** 2
print("""
      A_true = {0}
      A_est  = {1:.0f} +/- {2:.0f} (based on {3} measurements)
      """.format(A0, (w * Ai).sum() / w.sum(), w.sum() ** -0.5, N))
      
      
      
      
# Definition logarithmique des termes
def log_prior(theta):
    return 1  # prior non informatif

def log_likelihood(theta, Ai, ei):
    return -0.5 * np.sum(np.log(2 * np.pi * ei ** 2)
                         + (Ai - theta) ** 2 / ei ** 2)

def log_posterior(theta, Ai, ei):
    return log_prior(theta) + log_likelihood(theta, Ai, ei)


#Calcul la probabilité a posteriori p
A = np.linspace(975, 1020, num = 500) #Paramètre du modéle
logp=[] #initialisation du logp

for Avar in A:
    logp.append(log_posterior(Avar, Ai, ei))

#calcul de p et normalisation à l'unité   
from scipy.integrate import simps 
p = np.exp(logp)
p /=simps(p,A)

#tracé d
fig, ax = plt.subplots()
plt.plot(A,p)
plt.xlabel("Activité en coups par seconde"); plt.ylabel("p(A|D)")




from scipy.optimize import curve_fit
#from scipy import asarray as ar,exp
#import pylab as plb

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

popt,pcov = curve_fit(gaus,A,p,p0=[1,1000,10])

"""
fig, ax = plt.subplots()
plt.plot(A,gaus(A,*popt), color='r')
"""

ax.text(977,0.075,"$A_0 =$ {}\n".format(A0)+"$A_{crédible}$"+ "= {0:.0f}".format(popt[1])+" $\pm$"+" {0:.0f}".format(popt[2]),fontsize=10)
      
section = np.arange(popt[1]-popt[2], popt[1]+popt[2], 1/20.)
plt.fill_between(section,gaus(section,*popt), alpha = 0.4, color = 'gray')
ax.vlines([A0],0, np.max(p), linewidth=3, alpha=0.7, color = '#21177D')
import matplotlib.patches as patches
rect1 = patches.Rectangle((975,0.082),1,0.004,linewidth=1,edgecolor='none',facecolor='#21177D')
ax.add_patch(rect1)
rect1 = patches.Rectangle((975,0.075),1,0.004,linewidth=1,edgecolor='none',facecolor='gray')
ax.add_patch(rect1)
rect1 = patches.Rectangle((974,0.071),18,0.02,linewidth=1,edgecolor='k',facecolor='none')
ax.add_patch(rect1)
plt.savefig("C:/Users/Romain/Desktop/Mesures/Images/PosterieurSimpleActiviteRadioactive.pdf", bbox_inches='tight', transparent=True, pad_inches=0)



print("""
      A0 = {0}
      A_est  = {1:.0f} +/- {2:.0f} (based on {3} measurements)
      """.format(A0, popt[1], popt[2], N))