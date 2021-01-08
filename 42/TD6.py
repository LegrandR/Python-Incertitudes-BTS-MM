# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 11:06:36 2020

@author: Romain
"""


import random
print(random.randint(0,10))



import  math
print(math.cos(math.pi / 2))
print(math.sin(math.pi / 2))




from  random  import  randint
print(random.randint(0,10))




import  matplotlib.pyplot  as plt

temps = [1, 2, 3, 4, 6, 7, 9]
concentration = [5.5, 7.2, 11.8, 13.6, 19.1, 21.7,  29.4]
plt.scatter(temps , concentration , marker ="o", color ="blue")
plt.xlabel ("Temps (h)")

plt.ylabel (" Concentration (mg/L)")
plt.title(" Concentration  de  produit  en  fonction  du  temps")
plt.show()



import  numpy  as np
import  matplotlib.pyplot  as plt

temps = [1, 2, 3, 4, 6, 7, 9]
concentration = [5.5, 7.2, 11.8, 13.6, 19.1, 21.7,  29.4]

plt.scatter(temps , concentration , marker ="o", color ="blue")
plt.xlabel ("Temps (h)")
plt.ylabel (" Concentration (mg/L)")
plt.title(" Concentration  de  produit  en  fonction  du  temps")

x = np.linspace(min(temps), max(temps), 50)
y = 2 + 3 * x

plt.plot(x, y, color='green', ls="--")
plt.grid()

path = 'C:/SSD/Cours/Lycee/BTS TPIL/TPIL2019-2020/Python/6/Images/' #A modifier !!!
plt.savefig(path+'concentration_vs_temps.png', bbox_inches='tight', dpi = 200)


























