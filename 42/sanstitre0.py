# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 10:08:37 2020

@author: Romain
"""


import numpy as np
import matplotlib.pyplot as plt

NbP = 100

Nr = np.linspace(0,NbP,NbP)
p = 0.05
Nj = 5

f = p/(Nj-p*Nr+1) + p

fig, ax = plt.subplots()
plt.plot(Nr, f, '.')
