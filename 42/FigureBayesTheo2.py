# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:56:32 2020

@author: Romain
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 22:18:43 2020

@author: Romain
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats
from matplotlib.patches import PathPatch, Path


#Param
#Image size
xmax = 12
ymax = 3.3
scale=1

FigDecaleX = 0
FigDecaleY = 0

# Create figure and axes
fig, axs = plt.subplots(figsize=(xmax/2.54, ymax/2.54))

axs.get_xaxis().set_visible(False)
axs.get_yaxis().set_visible(False)
fig.patch.set_visible(False)

axs.set_xlim([0,xmax])
axs.set_ylim([0,ymax])
'''
#1
#Create a rectangle
decaleX = 1*scale
decaleY = 1*scale
decaleX +=FigDecaleX
decaleY +=FigDecaleY

rect1 = patches.Rectangle((0+decaleX,0+decaleY),3*scale,10*scale,linewidth=1,edgecolor='k',facecolor='#FDFC00')
rect2 = patches.Rectangle((3*scale+decaleX,0+decaleY),7*scale,10*scale,linewidth=1,edgecolor='k',facecolor='#6D9B53')


# Add the patch to the Axes
axs.add_patch(rect1)
axs.add_patch(rect2)

axs.text(5*scale+decaleX, -3*scale+0.1+ decaleY, 'Ensemble\ndes possibilités',horizontalalignment='center')
'''


#2
#Create a rectangle
decaleX = 1*scale
decaleY = 1*scale
decaleX +=FigDecaleX
decaleY +=FigDecaleY
decaleXY = [decaleX, decaleY]

Dim1 = [10,1]
Pos1 = [0,0]
Dim2 = [8.5,1]
Pos2 = [0,0]
Dim3 = [1.5,1]
Pos3 = [8.5,0]


rect1 = patches.Rectangle(np.add(Pos1*scale, decaleXY),Dim1[0]*scale,Dim1[1]*scale,linewidth=1,edgecolor='k',facecolor='none')
rect2 = patches.Rectangle(np.add(Pos2*scale , decaleXY),Dim2[0]*scale,Dim2[1]*scale,linewidth=1,edgecolor='k',facecolor='#63C2DC')
rect3 = patches.Rectangle(np.add(Pos3*scale , decaleXY),Dim3[0]*scale,Dim3[1]*scale,linewidth=1,edgecolor='k',facecolor='#877075')


# Add the patch to the Axes
axs.add_patch(rect1)
axs.add_patch(rect2)
axs.add_patch(rect3)
#axs.add_patch(rect4)

Pos1 = [4.5,1.5]
axs.text(Pos1[0]*scale + decaleX, Pos1[1]*scale + decaleY, 'Maths',horizontalalignment='center')
Pos1 = [4.5,0.3]
axs.text(Pos1[0]*scale + decaleX, Pos1[1]*scale + decaleY, '90%',horizontalalignment='center')
Pos1 = [9.2,1.5]
axs.text(Pos1[0]*scale + decaleX, Pos1[1]*scale + decaleY, 'Commerce',horizontalalignment='center')
Pos1 = [9.3,0.3]
axs.text(Pos1[0]*scale + decaleX, Pos1[1]*scale + decaleY, '10%',horizontalalignment='center')

plt.savefig("C:/Users/Romain/Desktop/Mesures/Images/ConceptionProbaExempleBayes.pdf", bbox_inches="tight")

'''
#3
#Create a rectangle
decaleX = 25*scale
decaleY = 1*scale
decaleX +=FigDecaleX
decaleY +=FigDecaleY

rect1 = patches.Rectangle((0+decaleX,0+decaleY),14*scale,10*scale,linewidth=1,edgecolor='k',facecolor='none', ls=":")
rect2 = patches.Rectangle((1*scale+decaleX,5*scale+decaleY),12*scale,0.1*scale,linewidth=1,edgecolor='k',facecolor='k')
rect3 = patches.Rectangle((6*scale+decaleX,5.5*scale+decaleY),3*scale,4*scale,linewidth=1,edgecolor='k',facecolor='#63C2DC')
rect3b = patches.Rectangle((1*scale+decaleX,0.5*scale+decaleY),3*scale,4*scale,linewidth=1,edgecolor='k',facecolor='#63C2DC')
rect4 = patches.Rectangle((6*scale+decaleX,2*scale+decaleY),7*scale,1*scale,linewidth=1,edgecolor='k',facecolor='#28758A')


# Add the patch to the Axes
axs.add_patch(rect1)
axs.add_patch(rect2)
axs.add_patch(rect3)
axs.add_patch(rect3b)
axs.add_patch(rect4)

axs.text(5*scale + decaleX, 2*scale+0.1 + decaleY, '+',horizontalalignment='center')

axs.text(7*scale + decaleX, -4*scale+0.1+ decaleY, 'Probabilité que Hypothése\nsachant Données',horizontalalignment='center')



'''

