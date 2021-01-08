# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:43:52 2020

@author: Romain
"""
"""
Convergence vers la loi normale de la moyenne d'un tirage aléatoire
continue et uniforme
"""

#Modules
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#Parametres 
np.random.seed(8)  #Pour la répétabilité du tirage aléatoire
P = 3000 # La moyenne est réalisée P fois
N =  5 # Nombres de mesure


#Partie principale du programme
x0 = (10*stats.uniform.rvs(0,1,N*P)) #Réalisation de P moyennes de N tirages

#Calcul de la moyenne suivant les N tirages
#X est un tableau contenant les P moyennes
X = np.mean(np.reshape(x0, [P,N]),1) 

# Ajustement par une courbe gaussienne
G_fit = np.linspace(0, 10, 1000)
pdf = stats.norm(np.mean(X), np.std(X)).pdf(G_fit)

#Tracé graphique
plt.xticks([0,2.5,5,7.5,10])
plt.hist(X, bins=101, histtype="stepfilled", alpha=0.3, density=True)
plt.xlim([-0.5,10.5])
plt.plot(G_fit, pdf, '-k', lw = 1)
plt.text(0,np.max(pdf)/1.1,'Moyenne : {0:.2f}\nEcart type : {1:.2f}'.format(np.mean(X),np.std(X)))


"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


#Parametres 
np.random.seed(8)  #


P = 300000 
N = 3
L = N*P
x0 = (10*stats.uniform.rvs(0,1,L))
xi = np.reshape(x0, [P,N] )
X = np.mean(xi, 1)
plt.yticks([])
plt.xticks([0,2.5,5,7.5,10])
plt.hist(X, bins=101, histtype="stepfilled", alpha=0.3, density=True)
#plt.set_xlim([-1,11])
 
 
LimBasse = 5 - 10/(np.sqrt(12*N))
LimHaute = 5 + 10/(np.sqrt(12*N))
 
#Pop.append(1-((X>LimHaute).sum() + (X<LimBasse).sum())/P)
# print("$P(\sigma)$ = {0:.2f}".format(1-((X>LimHaute).sum() + (X<LimBasse).sum())/P ))
 #print( 10/(np.sqrt(12*N)))
 #print(np.std(X))
 #print(np.mean(X))

 
 # plot a best-fit Gaussian
F_fit = np.linspace(0, 10, 1000)
pdf = stats.norm(np.mean(X), np.std(X)).pdf(F_fit)
plt.plot(F_fit, pdf, '-k', lw = 1)





plt.savefig("C:/Users/Romain/Desktop/Mesures/Images/TheoLimiteCentrale2.pdf", bbox_inches="tight")
plt.clf()



Ns = [1,2,3,4,5,6,7,8,10,20,40,80]
P = 300000      #Nombres d'étudiants passant le test

fig, axs = plt.subplots(3,4, sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0, 'wspace': 0})

index = 0
Pop = []

for i in range(3):
    for j in range(4):
       
        N= Ns[index]
        L = N*P
        index +=1
        x0 = (10*stats.uniform.rvs(0,1,L))
        xi = np.reshape(x0, [P,N] )
        X = np.mean(xi, 1)
        
        axs[i,j].hist(X, bins=101, histtype="stepfilled", alpha=0.3, density=True)
        axs[i,j].set_xlim([-1,11])
        
        
        LimBasse = 5 - 10/(np.sqrt(12*N))
        LimHaute = 5 + 10/(np.sqrt(12*N))
    
        Pop.append(1-((X>LimHaute).sum() + (X<LimBasse).sum())/P)
       # print("$P(\sigma)$ = {0:.2f}".format(1-((X>LimHaute).sum() + (X<LimBasse).sum())/P ))
        #print( 10/(np.sqrt(12*N)))
        #print(np.std(X))
        #print(np.mean(X))

        
        # plot a best-fit Gaussian
        F_fit = np.linspace(0, 10, 1000)
        pdf = stats.norm(np.mean(X), np.std(X)).pdf(F_fit)
        
        axs[i,j].plot(F_fit, pdf, '-k', lw = 0.2)
        axs[i,j].set_ylim([0,1.3*np.max(pdf)])
        axs[i,j].yaxis.set_ticks([])
        axs[i,j].xaxis.set_ticks([])
                
     
index = 0        
for i in range(3):
    for j in range(4):
        N= Ns[index]
        (ymin, ymax) = axs[i,j].get_ylim()
        axs[i,j].text(-0.5, 0.85*ymax,'$N = {}$'.format(N), size = 7)
        axs[i,j].text(-0.5, 0.70*ymax,'$P = {0:.2f}$'.format(Pop[index]), size = 7)
        index +=1

axs[2,0].xaxis.set_ticks(np.arange(0, 11, 5))

plt.savefig("C:/Users/Romain/Desktop/Mesures/Images/TheoLimiteCentrale1.pdf", bbox_inches="tight")
plt.clf()

"""