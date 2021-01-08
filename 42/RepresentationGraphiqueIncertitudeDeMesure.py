# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 10:13:30 2020

@author: Romain
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600

np.random.seed(1)


# Generate some synthetic data from the model.
N = 100
Sig1 = 3
Sig2 = Sig1
Sig3 = 10

decal2 = 15
GrapheRange=10

x1 = stats.norm.rvs(size = N, loc = 0, scale = Sig1)
y1 = stats.norm.rvs(size = N, loc = 0, scale = Sig1)
CenterX1 = np.mean(x1)
CenterY1 = np.mean(y1)
SigData1 = np.sqrt(np.var(x1) + np.var(y1))/1.414


x2 = stats.norm.rvs(size = N, loc = decal2, scale = Sig2)
y2 = stats.norm.rvs(size = N, loc = decal2, scale = Sig2)
CenterX2 = np.mean(x2)
CenterY2 = np.mean(y2)
SigData2 = np.sqrt(np.var(x2) + np.var(y2))/1.414

x3 = stats.norm.rvs(size = N, loc = 0, scale = Sig3)
y3 = stats.norm.rvs(size = N, loc = 0, scale = Sig3)
CenterX3 = np.mean(x3)
CenterY3 = np.mean(y3)
SigData3 = np.sqrt(np.var(x3) + np.var(y3))/1.414

plt.scatter(x1,y1, s = 0.3, marker = '.', color = 'r')
#circle1 = plt.Circle((0,0), Sig1, color='r', fill=False, lw=1, ls = ':')
circle11 = plt.Circle((CenterX1,CenterY1), SigData1, color='r', fill=False, lw=0.5, ls = '-')
#plt.gcf().gca().add_artist(circle1)
plt.gcf().gca().add_artist(circle11)

plt.scatter(x2,y2, s = 0.3, marker = '.', color = 'g')
#circle2 = plt.Circle((decal2,decal2), Sig2, color='g', fill=False, lw=1, ls = ':')
circle22 = plt.Circle((CenterX2,CenterY2), SigData2, color='g', fill=False, lw=0.5, ls = '-')
#plt.gcf().gca().add_artist(circle2)
plt.gcf().gca().add_artist(circle22)

plt.scatter(x3,y3, s = 0.3, marker = '.', color = 'b')
#circle3 = plt.Circle((0,0), Sig3, color='b', fill=False, lw=1, ls = ':')
circle33 = plt.Circle((CenterX3,CenterY3), SigData3, color='b', fill=False, lw=0.5, ls = '-')
#plt.gcf().gca().add_artist(circle3)
plt.gcf().gca().add_artist(circle33)

plt.axis("off")

plt.axis("scaled")
