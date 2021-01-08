# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 22:42:12 2020

@author: Romain
"""




# Data generation
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import rc
import emcee
from tqdm import tqdm
rc('text', usetex=True)

np.random.seed(5)  # for repeatability 5

a = 5      #light house position along the coast
b = 4      #light house distance from the coast

c_start = -1000  #coast start
c_end   =  1000  #coast end
b_max =20
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

N=1
c_pos=c_pos_full[0:N]


"""
#algorithme emce : détermination des paramètre a et b
ndim = 2  # number of parameters in the model
nwalkers = 200  # number of MCMC walkers
nburn = 1000  # "burn-in" period to let chains stabilize
nsteps = 2000  # number of MCMC steps to take"""

# Nombre de points
Nb_alpha = 100
Nb_beta = 100

# Création des tableaux pour mu et sigma
alpha_line = np.linspace(-50,50,Nb_alpha)
beta_line = np.linspace(0,b_max,Nb_beta)

# Création du maillage
alpha_mesh, beta_mesh = np.meshgrid(alpha_line, beta_line)

# Initialisation du log du postérieur
logP = np.zeros([Nb_alpha,Nb_beta])


def log_prior(alpha, beta):
    # sigma needs to be positive.
    if (beta < 0.001 or beta>b_max or alpha < c_start or alpha > c_end) :
        return -np.inf
    else:
        return 0

def log_likelihood(alpha, beta, c_pos):
    if (beta < 1 or beta>b_max or alpha < c_start or alpha > c_end) :
        return -np.inf
    else:
        return - np.sum(np.log((beta**2 + (c_pos - alpha)**2)/beta))

def log_posterior(alpha, beta, c_pos):
    return log_prior(alpha,beta) + log_likelihood(alpha,beta, c_pos)




# Calcul du postérieur avec double boucle
for i in tqdm(range(Nb_alpha)):
    var_alpha = alpha_line[i]
    for j in range(Nb_beta):
        var_beta = beta_line[j]
        logP[j,i] += log_posterior(var_alpha, var_beta, c_pos)

# Normalisation à 1
logP -= np.max(logP)
p = np.exp(logP)

# Valeur la plus probable
result = np.where(p == np.amax(p))

# Affichage graphique
fig, ax = plt.subplots()
ax.imshow(p, extent=[alpha_line.min(), alpha_line.max(), beta_line.min(), beta_line.max()], interpolation = 'bicubic')
levels = [0.68, 0.95]
CS = ax.contour(alpha_mesh, beta_mesh,p, levels, colors='k', linewidths = 0.7)
ax.plot(a, b, 'o', color='red', ms=5);
ax.text(alpha_line[result[1]][0],beta_line[result[0]][0], '+' )


fig, axs = plt.subplots(jmax,imax,   gridspec_kw={'hspace': 0.6, 'wspace': 0.25},figsize=(xmax/2.54, ymax/2.54))

#tracé
from astroML.plotting import plot_mcmc
for j in range(jmax):
    for i in range(imax):
        logP = np.zeros([Nb_alpha,Nb_beta])
        N = Ns[imax*j + i]
        c_pos=c_pos_full[0:N]
        # we'll start at random locations between 0 and 2000 and c_start and c_end /beta
        """starting_guesses = np.random.rand(nwalkers, ndim)    
        
        starting_guesses[:,0] = 40 *starting_guesses[:,0] -20
        starting_guesses[:,1] *= 20
        starting_guesses[:,1] += 5
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[c_pos])
        sampler.run_mcmc(starting_guesses, nsteps)
        
        sample = sampler.chain  # shape = (nwalkers, nsteps, ndim)
        sample = sampler.chain[:, nburn:, :].reshape(-1, 2)"""
        
        

        # Calcul du postérieur avec double boucle
        for ii in tqdm(range(Nb_alpha)):
            var_alpha = alpha_line[ii]
            for jj in range(Nb_beta):
                var_beta = beta_line[jj]
                logP[jj,ii] += log_posterior(var_alpha, var_beta, c_pos)
        
        # Normalisation à 1
        logP -= np.max(logP)
        p = np.exp(logP)
        
        # Valeur la plus probable
        result = np.where(p == np.max(p))

        # Affichage graphique
        axs[j,i].pcolor(alpha_mesh, beta_mesh, p)
        levels = [0.68, 0.95]
        CS = axs[j,i].contour(alpha_mesh, beta_mesh,p, levels, colors='k', linewidths = 0.7)
        #axs[j,i].text(alpha_line[result[1]][0],beta_line[result[0]][0], '+' )
        
        #axs[j,i] = plot_mcmc(sample.T, fig=fig, labels=[r'position $\alpha$ sur la côte', r'distance $\beta$ à la côte'], colors='k',  linewidths =1)
        axs[j,i].set_ylim(0, 20)
        axs[j,i].set_xlim(-50, 50)
       # axs[j,i].plot(sample[:, 0], sample[:, 1], ',k', alpha=0.02)
        axs[j,i].plot(a, b, 'o', color='red', ms=3);
        axs[j,i].set_xlabel(r"position $\alpha$ sur la côte")
        axs[j,i].set_ylabel(r"distance $\beta$ à la côte")
        fig.suptitle(r'$p(\alpha, \beta |\lbrace x_k \rbrace)$', fontsize=16)
        axs[j,i].set_title(r'$N=${}'.format(N))
        y_pos = -1.5*np.ones(N)
        if N<20:
            axs[j,i].scatter(c_pos, y_pos, marker = 'o', alpha=1, color = "none", edgecolor='k', s= 12)
            axs[j,i].set_ylim(-3, 20)
        

plt.savefig("C:/Users/Romain/Desktop/Mesures/Images/DeuxParaPharePosterieur.pdf", bbox_inches='tight', transparent=True, pad_inches=0.1)
