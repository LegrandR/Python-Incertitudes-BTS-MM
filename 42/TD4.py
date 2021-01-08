# -*- coding: utf-8 -*-
"""
Created on Sun May 17 14:52:24 2020

@author: Romain
"""

#1
x = 2
if x == 2:
    print('Le test est vrai !')
    
    
    
#2
x = 'souris'
if x == 'tigre':
    print('Le test est vrai !')
    
#3
x = 2
if x == 2:
    print('Le test est vrai !')
else:
    print('Le test est faux !')
    
x = 3
if x == 2:
    print('Le test est vrai !')
else:
    print('Le test est faux !')
        
    
#4
import random
Choix = random.choice([1, 2, 3, 4])

if Choix == 1:
    print('Vérification des multimètres')
elif Choix == 2:
    print('Vérification des bains thermostatés')
elif Choix == 3:
    print('Vérification des oscilloscopes')
elif Choix == 4:
    print('Vérification des clefs dynamométriques')
    
    
#5
nombres = [4,5,6]
for nb in nombres:
    if nb == 5 :
        print('Le test est vrai !')
        print('car la variable est {}'.format(nb))
        
for nb in nombres:
    if nb == 5 :
        print('Le test est vrai !')
    print('car la variable est {}'.format(nb))
    
    
#6
x = 2
y = 2
if x == 2 and y == 2:
    print('Le test est vrai')
    
    
#7
x = 2
y = 2
if x == 2 :
    if y == 2 :
        print('Le test est vrai')
        
#8
True or False
not True
not (True or False)


#9
for i in range(5):
    if i>2:
        break
    print(i)


#10
for i in range(5):
    if i == 2:
        continue
    print(i)


#11
1/10 == 0.1

#12
(3-2.7) == 0.3
3-2.7

#13
delta = 0.0001
var = 3 - 2.7
0.3 - delta < var < 0.3 + delta
abs(var - 0.3) < delta

#14
4%3

5%3

5%2

4%2

6%2

7%2

#15

[[48.6,  53.4],[-124.9, 156.7],[-66.2, -30.8], 
[-58.8, -43.1],[-73.9,  -40.6],[-53.7, -37.5], 
[-80.6, -26.0],[-68.5,  135.0],[-64.9, -23.5], 
[-66.9, -45.5],[-69.6,  -41.0],[-62.7, -37.5], 
[-68.2, -38.3],[-61.2,  -49.1],[-59.7, -41.1]]



print('[48.6,  53.4] n\'est pas en hélice')


#16
lettre = input("Entrer une lettre :")
print(lettre)


























    