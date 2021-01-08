# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 15:19:58 2020

@author: Romain
"""


# Création d'une série de données pour l'accélération de la pesanteur
import numpy as np
from scipy import stats
np.random.seed(123)  # Pour la répétabilité, les données sont aléatoires mais fixées avec ce paramètre

A0 = 10  # Valeur de référence de l'activité radioactive de l'échantillon en coups par seconde
N = 50 # Nombre de mesures
A = stats.poisson(A0).rvs(N)  # Réalisation des N mesures
e = np.sqrt(A)  # écart type estimé pour une loi de Poisson


