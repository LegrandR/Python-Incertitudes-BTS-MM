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
xmax = 19
ymax = 8
scale=0.47

FigDecaleX = 0
FigDecaleY = 2

# Create figure and axes
fig, axs = plt.subplots(figsize=(xmax/2.54, ymax/2.54),frameon=False)

axs.get_xaxis().set_visible(False)
axs.get_yaxis().set_visible(False)
fig.patch.set_visible(False)

axs.set_xlim([0,xmax])
axs.set_ylim([0,ymax])

#1
#Create a rectangle
decaleX = 1*scale
decaleY = 1*scale
decaleX +=FigDecaleX
decaleY +=FigDecaleY

rect1 = patches.Rectangle((0+decaleX,0+decaleY),1*scale,10*scale,linewidth=1,edgecolor='k',facecolor='#63C2DC', alpha =0.3)
rect2 = patches.Rectangle((1*scale+decaleX,0+decaleY),9*scale,10*scale,linewidth=1,edgecolor='k',facecolor='#877075', alpha =0.3)
rect3 = patches.Rectangle((1*scale+decaleX,0+decaleY),9*scale,10*scale,linewidth=1,edgecolor='k',facecolor='none', alpha =1)
rect4 = patches.Rectangle((0+decaleX,0+decaleY),1*scale,10*scale,linewidth=1,edgecolor='k',facecolor='none')
# Add the patch to the Axes
axs.add_patch(rect1)
axs.add_patch(rect2)
axs.add_patch(rect3)
axs.add_patch(rect4)
axs.text(5*scale+decaleX, -3*scale+0.1+ decaleY, 'Ensemble\ndes étudiants',horizontalalignment='center')
axs.text(0.58*scale+decaleX, 4*scale+0.1+ decaleY, 'Maths',horizontalalignment='center',rotation=90)
axs.text(5.5*scale+decaleX, 4*scale+0.1+ decaleY, 'Commerce',horizontalalignment='center',rotation=0)

axs.text(0.5*scale+decaleX, 10*scale+0.1+ decaleY, '1',horizontalalignment='center',rotation=0)
axs.text(5.5*scale+decaleX, 10*scale+0.1+ decaleY, '20',horizontalalignment='center',rotation=0)
#2
#Create a rectangle
decaleX = 13*scale
decaleY = 1*scale
decaleX +=FigDecaleX
decaleY +=FigDecaleY

rect1 = patches.Rectangle((0+decaleX,0+decaleY),1*scale,10*scale,linewidth=1,edgecolor='k',facecolor='none')
rect2 = patches.Rectangle((1*scale+decaleX,0+decaleY),9*scale,10*scale,linewidth=1,edgecolor='k',facecolor='none')
rect3 = patches.Rectangle((0+decaleX,0+decaleY),1*scale,4*scale,linewidth=1,edgecolor='k',facecolor='#63C2DC')
rect6 = patches.Rectangle((0+decaleX,0+decaleY),1*scale,10*scale,linewidth=1,edgecolor='k',facecolor='#63C2DC', alpha =0.3)
rect4 = patches.Rectangle((1*scale+decaleX,0+decaleY),9*scale,1*scale,linewidth=1,edgecolor='k',facecolor='#877075')
rect5 = patches.Rectangle((1*scale+decaleX,0+decaleY),9*scale,10*scale,linewidth=1,edgecolor='k',facecolor='#877075', alpha =0.3)


# Add the patch to the Axes
axs.add_patch(rect1)
axs.add_patch(rect5)
axs.add_patch(rect6)
axs.add_patch(rect2)
axs.add_patch(rect3)
axs.add_patch(rect4)


axs.text(5*scale + decaleX, -4*scale+0.1+ decaleY, 'Ensemble\ndes étudiants\nvérifiant la description',horizontalalignment='center')
axs.text(0.5*scale+decaleX, 10*scale+0.1+ decaleY, '40%',horizontalalignment='center',rotation=0)
axs.text(5.5*scale+decaleX, 10*scale+0.1+ decaleY, '5%',horizontalalignment='center',rotation=0)


#3
#Create a rectangle
decaleX = 25*scale
decaleY = 1*scale
decaleX +=FigDecaleX
decaleY +=FigDecaleY

rect1 = patches.Rectangle((0+decaleX,0+decaleY),14*scale,10*scale,linewidth=1,edgecolor='k',facecolor='none', ls=":")
rect2 = patches.Rectangle((1*scale+decaleX,5*scale+decaleY),12*scale,0.1*scale,linewidth=1,edgecolor='k',facecolor='k')
rect3 = patches.Rectangle((6*scale+decaleX,5.5*scale+decaleY),1*scale,4*scale,linewidth=1,edgecolor='k',facecolor='#63C2DC')
rect3b = patches.Rectangle((1*scale+decaleX,0.5*scale+decaleY),1*scale,4*scale,linewidth=1,edgecolor='k',facecolor='#63C2DC')
rect4 = patches.Rectangle((4*scale+decaleX,2*scale+decaleY),9*scale,1*scale,linewidth=1,edgecolor='k',facecolor='#877075')


# Add the patch to the Axes
axs.add_patch(rect1)
axs.add_patch(rect2)
axs.add_patch(rect3)
axs.add_patch(rect3b)
axs.add_patch(rect4)

axs.text(3*scale + decaleX, 2*scale+0.1 + decaleY, '+',horizontalalignment='center')

axs.text(7*scale + decaleX, -4*scale+0.1+ decaleY, '''Probabilité que l'étudiant\nsoit doctorant en maths\nsachant la description''',horizontalalignment='center')



plt.savefig("C:/Users/Romain/Desktop/Mesures/Images/ResolutionExempleBayes.pdf", bbox_inches='tight', transparent=True, pad_inches=-0.1)

