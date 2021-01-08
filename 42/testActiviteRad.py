# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 23:05:58 2020

@author: Romain
"""





# Création d'une série de données pour l'accélération de la pesanteur
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm

# Pour la répétabilité:
# les données sont aléatoires mais fixées avec ce paramètre
np.random.seed(42) 

N = 100 # Nombre de mesures
# Valeur de référence de l'activité radioactive
# de l'échantillon en coups par seconde et écart type du modèle
mu_A, sigma_A = 1000, 10  

A0 = stats.norm(mu_A, sigma_A).rvs(N) #Activité de référence fluctuante
Ai = stats.poisson(A0).rvs(N)  # Réalisation des N mesures
ei = np.sqrt(Ai)  # écart type estimé pour une loi de Poisson


      
# Definition logarithmique des termes
def log_prior(mu, sig):
        # theta[0] représente mu_A, et theta[1] est sigma_A
        # sigma_A DOIT être positif.
    if sig <= 0:
        return -np.inf
    else:
        return 0

#Log de la vraisemblance
def log_likelihood(mu, sig, Ai, ei):
    return -0.5 * np.sum(np.log(2 * np.pi * (sig ** 2 + ei ** 2))
                         + (Ai - mu) ** 2 / (sig ** 2 + ei ** 2))

#log du postérieur
def log_posterior(mu, sig, Ai, ei):
    return log_prior(mu, sig) + log_likelihood(mu, sig, Ai, ei)


# Calcul du postérieur

# Nombre de points
Nb_mu = 200
Nb_sig = 200

# Création des tableaux pour mu et sigma
mu_line = np.linspace(990,1010,Nb_mu)
sig_line = np.linspace(0,30,Nb_sig)

# Création du maillage
mu_mesh, sig_mesh = np.meshgrid(mu_line, sig_line)

# Initialisation du log du postérieur
logP = np.zeros([Nb_mu,Nb_sig])

# Calcul du postérieur avec double boucle
for i in tqdm(range(Nb_mu)):
    var_mu = mu_line[i]
    for j in range(Nb_sig):
        var_sig = sig_line[j]
        logP[j,i] += log_posterior(var_mu, var_sig, Ai, ei)
        
# Normalisation à 1
logP -= np.max(logP)
p = np.exp(logP)

# Valeur la plus probable
result = np.where(p == np.amax(p))

# Affichage graphique
fig, ax = plt.subplots()
ax.pcolor(mu_mesh, sig_mesh, p)
levels = [0.68, 0.95]
CS = ax.contour(mu_mesh, sig_mesh,p, levels, colors='k', linewidths = 0.7)
ax.plot([mu_A], [sigma_A], 'o', color='red', ms=4);
ax.text(mu_line[result[1]][0],sig_line[result[0]][0], '+' )


plt.savefig("C:/Users/Romain/Desktop/Mesures/Images/PosterieurDeuxParametresActiviteRadioactive.pdf", bbox_inches='tight', transparent=True, pad_inches=0)






print("Valeur de mu la plus probable : {:.1f}.".format(mu_line[result[1]][0]) )
print("Valeur de sig la plus probable : {:.1f}.".format(sig_line[result[0]][0]) )