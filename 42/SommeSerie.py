import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from math import pi 

def sf_triangle(t,n=10):
    i = 0
    S = 0
    while i < n:
        S = S + np.cos((2*i+1)*t)/((2*i+1)**2)
        i=i+1
    return 8/pi*S

def sf_dents(t,n=10):
    i = 1
    S = 0
    while i < n:
        S = S + ((-1)**(i-1))*np.sin(i*t)/i
        i=i+1
    return 2/pi*S

def dents(t):
   return (((t-pi)%(2*pi)) - pi)/pi

t = np.arange(0,100,.0001)

plt.plot(t,sf_dents(t,100), '.',t, dents(t))


A = np.fft.fft(dents(t))
# visualisation de A
# on ajoute a droite la valeur de gauche pour la periodicite
B = np.append(A, A[0])
plt.plot(np.abs(B[0:100]), '.')



plt.show()