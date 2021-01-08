# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 10:08:37 2020

@author: Romain
"""



import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

mpl.rcParams['figure.dpi'] = 600
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

np.random.seed(10)  # for repeatability

P = 15000
Nline = range(10,100,10)
#L = N*P
StdX=[]
sigma = 1


for N in tqdm(Nline):
    xmean = []
    for i in range(P):
        x0 = [] #[stats.norm.rvs(size = 1, loc = 0, scale = sigma)]  np.mean(x0)
        for j in range(N):
            #x0.append(stats.norm.rvs(size = 1, loc = 0, scale = sigma)) #+ centre[0]
            x0.append(stats.cauchy.rvs(size = 1, loc = 0, scale = sigma))
        xmean.append(np.mean(x0))
    StdX.append(np.std(xmean))

    

Y = 1/np.square(StdX)
plt.scatter(np.log(Nline), np.log(StdX))
#plt.yticks(np.log([0.08,0.2,0.4]))
plt.yticks([])
plt.xticks([2,3,4,5])
plt.ylabel("log($\sigma$)")
plt.xlabel("log(n)")
#plt.text(3.5,-1.6, r"Pente = -0.5")


fit = np.polyfit(np.log(Nline),np.log(StdX), 1 )
logx_fit = np.linspace(2, 5.0,1000)
logy_fit = fit[0]*logx_fit + fit[1]
plt.plot(logx_fit, logy_fit, "k", alpha=0.3)
plt.text(2,4, "$y=ax+b$\na = {0:.3f} \nb = {1:.3f} ".format(fit[0], fit[1]),fontsize=8)


#plt.savefig("C:/Users/Romain/Desktop/Mesures/Images/TheoLimiteCentraleCorrelated.pdf", bbox_inches="tight")
#plt.savefig("C:/Users/Romain/Desktop/Mesures/Images/TheoLimiteCentrale3.pdf", bbox_inches="tight")
plt.savefig("C:/Users/Romain/Desktop/Mesures/Images/TheoLimiteCentraleCauchy.pdf", bbox_inches="tight")

plt.hist(xmean, bins = 100)

"""
plt.yticks([])
#plt.xticks([0,2.5,5,7.5,10])
plt.hist(X, bins=range(-30,30), histtype="stepfilled", alpha=0.3, density=True)
#plt.set_xlim([-1,11])
print(np.std(X))
"""
#LimBasse = 5 - 10/(np.sqrt(12*N))
#LimHaute = 5 + 10/(np.sqrt(12*N))
 
#Pop.append(1-((X>LimHaute).sum() + (X<LimBasse).sum())/P)
# print("$P(\sigma)$ = {0:.2f}".format(1-((X>LimHaute).sum() + (X<LimBasse).sum())/P ))
 #print( 10/(np.sqrt(12*N)))
 #print(np.std(X))
 #print(np.mean(X))

""" 
 # plot a best-fit Gaussian
F_fit = np.linspace(0, 10, 1000)
pdf = stats.norm(np.mean(X), np.std(X)).pdf(F_fit)
plt.plot(F_fit, pdf, '-k', lw = 1)
"""
