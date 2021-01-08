# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:15:24 2020

@author: Romain
"""


# Exemple 1
filin = open("zoo.txt", "r")
lignes = filin.readlines ()
filin.close()
print(lignes)
for ligne in lignes : 
    print(ligne)


























    
    
with  open("zoo.txt",'r') as  filin:
    lignes = filin.readlines ()
   
print(lignes)
for ligne in lignes : 
    print(ligne)
    
    
    
    
with  open("zoo.txt", "r") as  filin:
    for  ligne  in  filin:
        print(ligne)
        
        
        
animaux = [" poisson", "abeille", "chat"]
with  open("zoo2.txt", "w") as  filout:
    for  animal  in  animaux:
        filout.write(animal)
        
        
        
        
animaux = [" poisson", "abeille", "chat"]
with  open("zoo2.txt", "w") as  filout:
    for  animal  in  animaux:
        filout.write ("{}\n". format(animal ))     
        
        
        
        