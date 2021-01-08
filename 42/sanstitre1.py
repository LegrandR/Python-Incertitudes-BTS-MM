# -*- coding: utf-8 -*-
"""
Created on Tue May  5 14:12:32 2020

@author: Romain
"""


fig, ax = plt.subplots()
plt.plot(a,b, 'ro')
plt.ylim(-5, 1.5*b)
plt.xlim(-100, 100)
plt.fill([1.2*c_start,1.2*c_start, 1.2*c_end, 1.2*c_end], [-5,0,0,-5], 'tab:gray', alpha=0.2)
ax.vlines(c_pos, 0, 0.3*b, linewidth=1, alpha=0.5)
ax.vlines(c_pos.mean(), 0, 1.1*c_end, linewidth=3, alpha=0.2, colors='red')