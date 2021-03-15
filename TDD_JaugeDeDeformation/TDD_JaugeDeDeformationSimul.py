# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 11:33:34 2021

@author: Prof
"""



import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import stats
import scipy.special
import statistics
from tqdm import tqdm


#Paramètre du tirage
N = 50000
sig = 0.05787
mu = 299.6938

Alea = stats.norm.rvs(size = N, loc = mu, scale = sig)
BaseTemps = np.linspace(0,N,N)

DATA = 5*sig*np.sin(40*BaseTemps/N) + Alea

print(np.mean(DATA))
print(np.std(DATA))

matplotlib.rcParams.update({'font.size': 30})

plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
plt.hist(DATA, bins=100, histtype="stepfilled", alpha=0.3, density=True)
plt.xlabel("x")
plt.ylabel("fréquence");


plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
F_fit = np.linspace(299.45,299.905,1000)
pdf = stats.norm(np.mean(DATA), np.std(DATA)).pdf(F_fit)
plt.plot(F_fit, pdf)
plt.xlabel("x")


Nmoy = 1
DATA2 = []
Ecart = [];

npoints = 50

for ind in tqdm(range(1,npoints)):
    Nmoy = ind
    DATA2 = []
    for i in range(len(DATA)): #for i in range(0,len(DATA),Nmoy):
        DATA2.append(np.mean(DATA[i:i+Nmoy]))
    Ecart.append(np.std(DATA2))    
   
"""    
print(np.mean(DATA2))
print(np.std(DATA2))
"""

Nmoy =50
DATA2 = []
for i in range(len(DATA)): #for i in range(0,len(DATA),Nmoy):
    DATA2.append(np.mean(DATA[i:i+Nmoy]))


plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
plt.hist(DATA2, bins=100, histtype="stepfilled", alpha=0.3, density=True)
#F_fit = np.linspace(299.0,300,1000)
#pdf = stats.norm(np.mean(DATA2), np.std(DATA2)).pdf(F_fit)
#plt.plot(F_fit, pdf)


plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
plt.plot( range(1,npoints),Ecart, "o", markersize=10, label = "Données corrélées")
plt.xlabel("Nombre de mesures")
plt.ylabel("écart-type de la moyenne")
plt.ylim([0, 0.225])
plt.legend()



matplotlib.rcParams.update({'font.size': 30})

plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
plt.plot(DATA, "o")
plt.xlabel("numéro de la mesure")
plt.ylabel("masse (g)")
