# -*- coding: utf-8 -*-
"""
Created on Wed May  6 20:31:15 2020

@author: Romain
"""

#ex
instruments = ['multimetre','oscilloscope','amperemetre','AD620']

for instru in instruments:
    print(instru)
    
    
#ex    
entiers = list(range(2,22,2))

p = 1
for n in entiers:
    p *=n
    print(p)
    
    
#ex
entiers = list(range(10))
ligne = ''
for i in entiers:
    ligne +='*'
    print(ligne)
    
#ex
entiers = list(range(10))
n = len(entiers)
etoile = '*'
for i in entiers:
    print(etoile*(n-i)  )
    
 
#ex
entiers = list(range(10))
n = len(entiers)
espace = ' '
etoile = '*'
for i in entiers:
    print(espace*(n-i) + etoile*i)    
    
    
    
#Pyramide
n=27
impairs = list(range(1,n,2))
etoile = '*'
for i in impairs:
    print('{:^{prec}s}'.format(etoile*i, prec = n)) 
    
    
#saut de puce
import random
import matplotlib.pyplot as plt

rep = 1000
Xlim = range(20)
Moyenne = []

for x in Xlim:
    ListCount = []
    for i in range(rep):
        count = 0
        position = 0
        while (position != x and position != -x):
            position += random.choice([-1,1])
            count +=1
        ListCount.append(count)        
    Moyenne.append(sum(ListCount)/len(ListCount))        
    print(Moyenne[-1])
    
plt.plot(Xlim, Moyenne)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    