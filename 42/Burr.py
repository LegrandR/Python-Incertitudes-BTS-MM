# -*- coding: utf-8 -*-
"""
Created on Tue May 26 09:57:11 2020

@author: Romain
"""

#librairy
import numpy as np





#Pieces
#Pour chaque piéce, on défini les coordonnées des emplacements occupés

A = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],
              [1,0,0],[1,1,0],
              [2,0,0],[2,1,0],
              [3,0,0],[3,0,1],[3,1,0],[3,1,1],])



#Burr
#Matrice 4x4x4 représentant la structure 3D du noyau du cube : 1 occupé, 0 vide 

Burr = np.array(
    [
     [
      [0,0,0,0],
      [0,0,0,0],
      [0,0,0,0],
      [0,0,0,0],
     ],
     [
      [0,0,0,0],
      [0,0,0,0],
      [0,0,0,0],
      [0,0,0,0],
     ],
     [
      [0,0,0,0],
      [0,0,0,0],
      [0,0,0,0],
      [0,0,0,0],
     ],
     [
      [0,0,0,0],
      [0,0,0,0],
      [0,0,0,0],
      [0,0,0,0],
     ],
    ])


#Rotation d'une piéce sur son axe
def rotation(P, n):
    n = n%4
    
    return P
    
    