# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 21:43:26 2021

@author: Prof
"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scipy.special
import matplotlib
from tqdm import tqdm

data = pd.read_csv("C:/Users/Prof/Desktop/PythonProjet/PNF_TraitementDeDonnees/TDD_JaugeDeDeformation/masse_50k.csv") 


data = data.values.tolist()


DATA = []

for x in data:
    DATA.append(x[0])
    

DATA = np.array(DATA)

#Paramètre du tirage
N = 10000
sig = 0.05787
mu = 299.6938

Alea = stats.norm.rvs(size = N, loc = mu, scale = sig)

#DATA = Alea
print(np.mean(DATA))
print(np.std(DATA))

     
    
#Rejet des valeurs abérrantes par écarts normalisés
def Rejet_Valeurs_Aberrantes(data, m=2.):
    d = np.abs(data - np.mean(data))
    mdev = np.std(d)
    s = d / (mdev if mdev else 1.)
    return data[s < m]


DATA = Rejet_Valeurs_Aberrantes(DATA, m = 5)    
       
    
    
#print(np.mean(DATA))
#print(np.std(DATA))

plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
plt.hist(DATA, bins=100, histtype="stepfilled", alpha=0.3, density=True)
F_fit = np.linspace(299.4,300,1000)
pdf = stats.norm(np.mean(DATA), np.std(DATA)).pdf(F_fit)
plt.xlabel("Masse (g)")
plt.ylabel("Fréquence")
plt.plot(F_fit, pdf)


Ecart = [];
Ecart2 = [];
npoints = 50

for ind in tqdm(range(1,npoints)):
    Nmoy = ind
    DATA2 = []
    for i in range(0,len(DATA),1):
        DATA2.append(np.mean(DATA[i:i+Nmoy]))
    Ecart2.append(np.std(DATA2))
    
    
print(np.mean(DATA2))
print(np.std(DATA2))

plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
plt.hist(DATA2, bins=100, histtype="stepfilled", alpha=0.3, density=True)
F_fit = np.linspace(299.4,300,1000)
pdf = stats.norm(np.mean(DATA2), np.std(DATA2)).pdf(F_fit)
plt.plot(F_fit, pdf)

plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
plt.plot( range(1,npoints),Ecart, "o", markersize=10, label = "Données corrélées")
plt.plot( range(1,npoints),Ecart2, "+", markersize=10, label = "Données non corrélées")
plt.xlabel("Nombre de mesures")
plt.ylabel("écart-type de la moyenne")
plt.legend()


matplotlib.rcParams.update({'font.size': 30})

plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
plt.plot(DATA, "o")
plt.ylim(290,310)
plt.xlabel("numéro de la mesure")
plt.ylabel("masse (g)")



for ind in range(1,npoints):
    Nmoy = ind
    DATA2 = []
    for i in range(0,len(DATA),1):
        DATA2.append(np.mean(DATA[i:i+Nmoy]))
    Ecart.append(np.std(DATA2))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





