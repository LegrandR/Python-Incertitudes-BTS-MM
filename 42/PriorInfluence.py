# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 08:38:17 2020

@author: Romain
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
#from tqdm import tqdm
from matplotlib import rc

np.random.seed(10) #10
#Nombre de tirage
Ptirage = 1000

#Nombre de mesure : 
Ns = [0,1,2,3,4,50,100,1000]

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

I_tot_mes = stats.norm.rvs(size = Ptirage, loc = I0, scale = sigmaI0)
U_tot_mes = stats.norm.rvs(size = Ptirage, loc = U0, scale = sigmaU0)



sigma_R = 5
sigma_prior = 2



def log_prior_carre(theta):
    #g_i needs to be between 0 and 1
    if (theta < R_nom*(1+Rtol)) and (theta > R_nom*(1-Rtol)):
        return 0
    else:
        return -np.inf  # recall log(0) = -inf

    
    
def log_prior_gauss(theta):
    return -((theta-490)**2)/(2*sigma_prior**2)

def log_likelihood(theta, R_mes):
    return - np.sum(((theta-R_mes)**2)/(2*sigma_R**2))


def log_posterior(theta, R_mes, logPrior):
    return logPrior + log_likelihood(theta, R_mes)



R_line = np.linspace(470, 535, 200)


#Graphique
imax = 2
jmax = 4
xmax = 20
ymax = 27
fig, axs = plt.subplots(jmax,imax,   gridspec_kw={'hspace': 0.59, 'wspace': 0.35},figsize=(xmax/2.54, ymax/2.54))

for j in range(jmax):
    for i in range(imax):
        N = Ns[imax*j + i]
        I_mes = I_tot_mes[0:N]
        U_mes = U_tot_mes[0:N]
        R_mes = U_mes / I_mes
        logp1 = []
        logp2 = []
        for R in R_line:
            logp1.append(log_posterior(R, R_mes, log_prior_gauss(R)))
            logp2.append(log_posterior(R, R_mes, log_prior_carre(R)))
            
        logp1 -= np.max(logp1)
        logp2 -= np.max(logp2)
        p1 = np.exp(logp1)
        p1 /=np.trapz(p1, R_line)
        p2 = np.exp(logp2)
        p2 /=np.trapz(p2, R_line)
        p1max = np.max(p1)      
        p2max = np.max(p2)      
        pmax = np.max([p1max, p2max])
        print(pmax)
        
        axs[j,i].plot(R_line, p1, 'k', ls = 'dotted', lw = 1)
        axs[j,i].plot(R_line, p2, 'b', ls = 'dashdot', lw = 1)
        axs[j,i].text(470,0.8*pmax,'N={}'.format(N))
        axs[j,i].text(468,1.1*pmax,'$R(p_{max})$' + '={:.1f}'.format(np.trapz(R_line*p1, R_line)),color='k')
        axs[j,i].text(509,1.1*pmax,'$R(p_{max})$' +'={:.1f}'.format(np.trapz(R_line*p2, R_line)+0.07),color='B')
        axs[j,i].set_xlabel(r"$R_0 (\Omega$)")
        axs[j,i].set_ylabel(r"$p(R_0 |\lbrace R_i \rbrace, \sigma_R)$")
        
        
plt.savefig("C:/Users/Romain/Desktop/Mesures/Images/PriorInfluence.pdf", bbox_inches='tight', transparent=True, pad_inches=0.5)