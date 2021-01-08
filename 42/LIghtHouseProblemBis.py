# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 21:46:04 2020

@author: Romain
"""




# Data generation
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)

np.random.seed(5)  # for repeatability 5

a = 5      #light house position along the coast
b = 4      #light house distance from the coast

c_start = -1000  #coast start
c_end   =  1000  #coast end
b_max = 50
t_start = np.arctan(-(a-c_start)/b)
t_end   = np.arctan((a+c_end)/b)

P =1000     #number of measurements : monter jusqu'à 1000 avec seed 5

theta = np.random.uniform(t_start, t_end, P)
# [-0.5, 1.2, 1.5]
c_pos_full = a + b * (np.tan(theta))

c_pos_full[2] -= 5
c_pos_full[4] = -120
c_pos_full[10] = -200
c_pos_full[80] = 5000

Ns = [1,2,3,5,7,15,20,100]
imax = 2
jmax = 4

xmax = 20
ymax = 27
fig, axs = plt.subplots(jmax,imax,   gridspec_kw={'hspace': 0.5, 'wspace': 0.15},figsize=(xmax/2.54, ymax/2.54))

#N=1
#c_pos = c_pos_full[0:N]

# Moyenne des angles
#print(theta.mean())

# Affichage du graphe , sharex='col', sharey='row',   gridspec_kw={'hspace': 0, 'wspace': 0}



def ProbPos(x, b, c_pos):
    P = 1
    for c in c_pos:
        P *= b / (3.14159 * (b ** 2 + (x - c)**2))
        P /= P.max()  #Pour normaliser
    return P
   
     
x=np.linspace(c_start,c_end,10000)


for j in range(jmax):
    for i in range(imax):
        N = Ns[imax*j + i]
        c_pos = c_pos_full[0:N]
        dpf = ProbPos(x, b, c_pos)
        
        
        axs[j,i].plot(x, dpf)
        axs[j,i].set_xlim(-50, 50)

        axs[j,i].set_yticks([])
        axs[j,i].set_ylim(0, 1.2)
        y_pos = 1.1*np.ones(N)
        if N<20:
            axs[j,i].scatter(c_pos, y_pos, marker = 'o', alpha=1, color = "none", edgecolor='k', s= 12)
        axs[j,i].vlines(c_pos.mean(), 0, 1, linewidth=2, alpha=0.3, colors='red')
        axs[j,i].text(-45,1,'N={}'.format(N))
        axs[j,i].text(-45,.85,'$\mu$={:.2f}'.format(np.mean(c_pos)))
        axs[j,i].set_xlabel(r"Position $\alpha$ du phare (km)")
        axs[j,i].set_ylabel(r"$p(\alpha |\lbrace x_k \rbrace, \beta)$")
        #axs[j,i].get_yaxis().set_visible(False)
        print("La valeur la plus probable de la position du phare est {:.2f}".format(x[np.argmax(dpf)]))


plt.savefig("C:/Users/Romain/Desktop/Mesures/Images/PharePosterieur.pdf", bbox_inches='tight', transparent=True, pad_inches=0)



print("La valeur la plus probable de la position du phare est {:.2f}".format(x[np.argmax(dpf)]))
print("La valeur moyenne de la position des données est {:.2f}".format(c_pos.mean()))


