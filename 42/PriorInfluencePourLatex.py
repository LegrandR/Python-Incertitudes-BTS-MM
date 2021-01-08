# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:22:58 2020

@author: Romain
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 08:38:17 2020

@author: Romain
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Pour la répétabilité
np.random.seed(10) 

#Nombre de mesures : 
N =1

#Valeur réelle de la résistance en Ohm
R_vrai = 512   

#Etude d'une résistance nominal de 500 Ohm à 5%
R_nom = 500
Rtol = 0.03

#Mesure : méthode voltampéremétrique 
#I = 1mA fixée et connue à 0.1mA
#U mesuré à 1mV
#Incertitude gaussienne, incertitude = agilent multimeter 34401a

I0 = 1  #mA
sigmaI0 = 0.01 #mA  résistance de shunt = 0.1 Ohm

U0 = 512  #mV
sigmaU0 = 0.002 #mV

I_mes = stats.norm.rvs(size = N, loc = I0, scale = sigmaI0)
U_mes = stats.norm.rvs(size = N, loc = U0, scale = sigmaU0)
R_mes = U_mes / I_mes
sigma_R = 5


# Définition des probabilités pour inférence bayésienne
def log_prior_uniforme(theta):
    #g_i needs to be between 0 and 1
    if (theta < R_nom*(1+Rtol)) and (theta > R_nom*(1-Rtol)):
        return 0
    else:
        return -np.inf  # recall log(0) = -inf


def log_likelihood(theta):
    return - np.sum(((theta-R_mes)**2)/(2*sigma_R**2))


def log_posterior(theta):
    return log_prior_uniforme(theta) + log_likelihood(theta)


# Calcul de la probabilité postérieur post
R_line = np.linspace(470, 535, 200)
logpost = [] #initialisation du logarithme du postérieur
for R in R_line:
    logpost.append(log_posterior(R))
post = np.exp(logpost)
post /=np.trapz(post, R_line)

# Tracé graphique
plt.plot(R_line, post)



