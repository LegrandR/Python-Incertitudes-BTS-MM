# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:17:41 2020

@author: Romain
"""


#exercice 1
ListeInstruments = ['multimetre','oscilloscope','amperemetre','AD620']

#solution 1 :
for instrument in ListeInstruments:
    print(instrument)
    
#solution 2 : 
for i in range(4):
    print(ListeInstruments[i])    

#solution 3 :
i = 0
imax = len(ListeInstruments)

while i < imax :
    print(ListeInstruments[i])
    i += 1  # i = i + 1
    
    
#Exercice 4
impairs = [1,3,5,7,9,11,13,15,17,19,21]
pairs = []

for NombreImpair in impairs:
    pairs.append(NombreImpair + 1)

print(pairs)




#Exercice 6 : 
Entiers = list(range(2, 21, 2))


#Exercice 9
import random

print(random.choice([-1,1]))
