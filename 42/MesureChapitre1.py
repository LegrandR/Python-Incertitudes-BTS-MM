# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 10:14:15 2020

@author: Romain
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import rc

moy = 50
EcType = 10

# plot a best-fit Gaussian
F_fit = np.linspace(0, 100, 100)
pdf = stats.norm(moy, EcType).pdf(F_fit)


fig, ax = plt.subplots()

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.plot(F_fit, pdf, '-k')
plt.xlabel("x"); plt.ylabel(r'$\mathcal{N}(x)$')
ax.vlines([moy], 0, 1*max(pdf), linewidth=2, alpha=0.4, color = 'b')
ax.text(moy+2,1*max(pdf), r'Moyenne $\mu$ de la loi')

ax.vlines([moy+EcType], 0, 0.6*max(pdf), linewidth=2, alpha=0.4, color = 'r')
ax.vlines([moy-EcType], 0, 0.6*max(pdf), linewidth=2, alpha=0.4, color = 'r')
arr_width = .009
#ax.arrow(moy-EcType, 0.005,2*EcType, 0,width =0.0001, head_width=0.001, head_length=1, fc='k', ec='k',length_includes_head=True)
ax.annotate('', xy=(moy-EcType,0.005), xytext=(moy+EcType,0.005),
            arrowprops={'arrowstyle': '<->'}, va='center')
ax.text(moy-8,0.006, r'Écart type')
fig.suptitle('Loi normale $\mathcal{N}(x)$')
