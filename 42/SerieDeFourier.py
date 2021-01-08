# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:06:26 2020

@author: Romain
"""


from numpy.fft import fft, fftfreq
import numpy as np
import math
from matplotlib.pyplot import figure, plot, xlabel, ylabel, vlines, axis, grid


T = 1.0
w0 = 2*math.pi/T
def signal(t):
    return 2.0*np.cos(w0*t)
    
    
#fe = 50.0
#te = 1/fe
N=15
Tf = 10
te = Tf / N
temps = np.linspace(0,Tf-te, N)  # np.arange(start=0.0,stop=10*T,step=te)
echantillons = signal(temps)

figure(figsize=(8,4))
plot(temps,echantillons,'r.')
xlabel('t')
ylabel('s')


#N = temps.size
tfd = fft(echantillons)/N
freq = fftfreq(N,te)

#freq = np.zeros(N)
#for k in range(N): 
 #   freq[k] = k*1.0/(10*T)
    
    
    
spectre = np.absolute(tfd)
figure(figsize=(10,4))
vlines(freq,[0],spectre,'r')
xlabel('f')
ylabel('S')
axis([-3,3,0,1.1*spectre.max()])
grid()
         