# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 18:47:34 2020

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
from scipy import interpolate

#Param
#Image size
xmax = 18
ymax = 13.5
scale=1

FigDecaleX = 4
FigDecaleY = 1

# Create figure and axes
fig, axs = plt.subplots(figsize=(xmax/2.54, ymax/2.54),frameon=False)

axs.get_xaxis().set_visible(False)
axs.get_yaxis().set_visible(False)
fig.patch.set_visible(False)

axs.set_xlim([0,xmax])
axs.set_ylim([0,ymax])

#2
#Create a rectangle
decaleX = 0*scale
decaleY = 0*scale
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


#haut
axs.text(0.5*scale+decaleX, 11*scale+0.1+ decaleY, '$p(H)$',horizontalalignment='center',rotation=0)
axs.text(5.5*scale+decaleX, 11*scale+0.1+ decaleY, r'$ p (\neg H ) $',horizontalalignment='center',rotation=0)


nodes = np.array([[decaleX,10.1*scale + decaleY],[decaleX,10.1*scale + decaleY+0.1],[0.5*scale+decaleX,10.9*scale+ decaleY-0.1],[0.5*scale+decaleX,10.9*scale+ decaleY]])
x = nodes[:,0]
y = nodes[:,1]
tck,u     = interpolate.splprep( [x,y] ,s = 0 )
xnew,ynew = interpolate.splev( np.linspace( 0, 1, 100 ), tck,der = 0)
plt.plot(xnew ,ynew, 'k' )

nodes = np.array([[1*scale + decaleX,10.1*scale + decaleY],[1*scale + decaleX,10.1*scale + decaleY+0.1],[0.5*scale+decaleX,10.9*scale+ decaleY-0.1],[0.5*scale+decaleX,10.9*scale+ decaleY]])
x = nodes[:,0]
y = nodes[:,1]
tck,u     = interpolate.splprep( [x,y] ,s = 0 )
xnew,ynew = interpolate.splev( np.linspace( 0, 1, 100 ), tck,der = 0)
plt.plot(xnew ,ynew,'k' )

nodes = np.array([[1*scale + decaleX+0.1,10.1*scale + decaleY],[1*scale + decaleX+0.1,10.1*scale + decaleY+0.00000001],[5.5*scale+decaleX,10.9*scale+ decaleY-0.1],[5.5*scale+decaleX,10.9*scale+ decaleY]])
x = nodes[:,0]
y = nodes[:,1]
tck,u     = interpolate.splprep( [x,y] ,s = 0 )
xnew,ynew = interpolate.splev( np.linspace( 0, 1, 100 ), tck,der = 0)
plt.plot(xnew ,ynew,'k' )

nodes = np.array([[10*scale + decaleX-0.1,10.1*scale + decaleY],[10*scale + decaleX-0.1,10.1*scale + decaleY+0.00000001],[5.5*scale+decaleX,10.9*scale+ decaleY-0.1],[5.5*scale+decaleX,10.9*scale+ decaleY]])
x = nodes[:,0]
y = nodes[:,1]
tck,u     = interpolate.splprep( [x,y] ,s = 0 )
xnew,ynew = interpolate.splev( np.linspace( 0, 1, 100 ), tck,der = 0)
plt.plot(xnew ,ynew,'k' )


#side

axs.text(-2*scale+decaleX-0.1, 2*scale+ decaleY-0.1, '$p(D|H)$',horizontalalignment='center',rotation=0)
axs.text(12*scale+decaleX+0.1, 0.5*scale + decaleY-0.1, r'$ p (D | \neg H ) $',horizontalalignment='center',rotation=0)


nodes = np.array([[decaleX,0*scale + decaleY],[decaleX-0.1,0*scale + decaleY],[-1.2*scale+decaleX+0.1, 2*scale+ decaleY],[-1.2*scale+decaleX, 2*scale+ decaleY]])
x = nodes[:,0]
y = nodes[:,1]
tck,u     = interpolate.splprep( [x,y] ,s = 0 )
xnew,ynew = interpolate.splev( np.linspace( 0, 1, 100 ), tck,der = 0)
plt.plot(xnew ,ynew, 'k' )

nodes = np.array([[decaleX,4*scale + decaleY],[decaleX-0.1,4*scale + decaleY],[-1.2*scale+decaleX+0.1, 2*scale+ decaleY],[-1.2*scale+decaleX, 2*scale+ decaleY]])
x = nodes[:,0]
y = nodes[:,1]
tck,u     = interpolate.splprep( [x,y] ,s = 0 )
xnew,ynew = interpolate.splev( np.linspace( 0, 1, 100 ), tck,der = 0)
plt.plot(xnew ,ynew,'k' )

nodes = np.array([[10*scale + decaleX,0*scale + decaleY],[10*scale + decaleX+0.1,0*scale + decaleY],[11*scale+decaleX-0.1, 0.5*scale+ decaleY],[11*scale+decaleX, 0.5*scale+ decaleY]])
x = nodes[:,0]
y = nodes[:,1]
tck,u     = interpolate.splprep( [x,y] ,s = 0 )
xnew,ynew = interpolate.splev( np.linspace( 0, 1, 100 ), tck,der = 0)
plt.plot(xnew ,ynew,'k' )

nodes = np.array([[10*scale + decaleX,1*scale + decaleY],[10*scale + decaleX+0.1,1*scale + decaleY],[11*scale+decaleX-0.1, 0.5*scale+ decaleY],[11*scale+decaleX, 0.5*scale+ decaleY]])
x = nodes[:,0]
y = nodes[:,1]
tck,u     = interpolate.splprep( [x,y] ,s = 0 )
xnew,ynew = interpolate.splev( np.linspace( 0, 1, 100 ), tck,der = 0)
plt.plot(xnew ,ynew,'k' )

plt.savefig("C:/Users/Romain/Desktop/Mesures/Images/PriorLikelyhoodAndEvidence.pdf", bbox_inches='tight', transparent=True, pad_inches=-0.1)

