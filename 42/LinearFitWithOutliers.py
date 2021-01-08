# -*- coding: utf-8 -*-
"""
Created on Fri May  8 22:12:37 2020

@author: Romain
"""


"""

x = np.array([ 0,  3,  9, 14, 15, 19, 20, 21, 30, 35,
              40, 41, 42, 43, 54, 56, 67, 69, 72, 88])
y = np.array([33, 68, 34, 34, 37, 71, 37, 44, 48, 49,
              53, 49, 50, 48, 56, 60, 61, 63, 44, 71])
e = np.array([ 3.6, 3.9, 2.6, 3.4, 3.8, 3.8, 2.2, 2.1, 2.3, 3.8,
               2.2, 2.8, 3.9, 3.1, 3.4, 2.6, 3.4, 3.7, 2.0, 3.5])


"""
"""
#Figure 0 : résidus données sur droite

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


np.random.seed(18)

# Choose the "true" parameters.
a_true = 2
b_true = -5
sigma = 5


# Generate some synthetic data from the model.
N = 10
x = 0.5 + np.sort(9 * np.random.rand(N))
y = a_true * x + b_true

yerr = stats.norm.rvs(size = N, loc = 0, scale = sigma)
y+= yerr


x0 = np.linspace(0, 10, 500)




for i in range(N):
    plt.plot([x[i],x[i]], [y[i],a_true * x[i] + b_true], "g", lw=1)

plt.plot(x0, a_true * x0 + b_true, "grey", lw=1.5, label="Loi initiale", zorder=0)
plt.scatter(x, y, label="Données mesurées", zorder=2)
i=0
plt.plot([x[i],x[i]], [y[i],a_true * x[i] + b_true], "g", lw=1, label="Erreurs de mesure $\epsilon_i$", zorder=0)

#plt.plot(x0, np.dot(np.vander(x0, 2), w), "-k", label="LS", lw=1)
plt.legend(fontsize=12)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y");

plt.scatter(x, a_true * x + b_true, color = "k", s=6, zorder=2 )

plt.annotate(r"$\epsilon_i$",fontsize=12,
            xy=(9,16.5)
            )

"""

"""
#Exemple 1 : Incertitudes constantes suivant y

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(17)

# Choose the "true" parameters.
a_true = 2
b_true = -5
sigma = 2

# Generate some synthetic data from the model.
Nb_test = 10000
N = 20
x = 0.5 + np.sort(9 * np.random.rand(N))
#yerr = 0.2 #+ 0.2 * np.random.rand(N)
y = a_true * x + b_true
yerr = stats.norm.rvs(size = N, loc = 0, scale = sigma)
y+= yerr
x0 = np.linspace(0, 10, 500)


#fiting
sigma_0 = sigma * np.ones(N)
A = np.vander(x, 2)
C = np.diag(sigma_0 * sigma_0)
ATA = np.dot(A.T, A / (sigma_0 ** 2)[:, None])
cov = np.linalg.inv(ATA)
w = np.linalg.solve(ATA, np.dot(A.T, y / (sigma) ** 2))

#figure
fig, axs = plt.subplots(1,2, gridspec_kw={ 'wspace': 0.35})
fig.set_size_inches(18/2.54, 7/2.54)
axs[0].errorbar(x, y, yerr=sigma, fmt=".k", capsize=0)
axs[0].plot(x0, a_true * x0 + b_true, color = "#2e538e", alpha=0.8, lw=3, label="Loi initiale")
axs[0].plot(x0, np.dot(np.vander(x0, 2), w), "--k", label="Ajustement linéaire")
axs[0].legend(fontsize=12)
axs[0].legend(loc = 2)
axs[0].set_xlim(0, 10)
axs[0].set_xlabel("x")
axs[0].set_ylabel("y");
axs[0].text(5.5,-7, "Estimateurs:\na = {0:.2f} ± {1:.2f} \nb = {2:.2f} ± {3:.2f}".format(w[0], np.sqrt(cov[0, 0]),w[1], np.sqrt(cov[1, 1])),fontsize=8)
axs[0].text(0.5,6, "$y=ax+b$\na = {0:.0f} \nb = {1:.0f} ".format(a_true, b_true),fontsize=8)
axs[0].set_ylim(-10, 26)
axs[0].xaxis.set_tick_params(labelsize=8)
axs[0].yaxis.set_tick_params(labelsize=8)

#Stats
a_calc = []
b_calc = []
std_a_calc=[]


for i in range(Nb_test):
    
    # Generate some synthetic data from the model.
    x = np.sort(10 * np.random.rand(N))
    sigma_y = 0.1 + 0.4 * np.random.rand(N)
    y = a_true * x + b_true
    yerr = stats.norm.rvs(size = N, loc = 0, scale = sigma)
    y+= yerr
    Nrand = np.random.randint(0,N-1)
    ymin = min(y)
    ymax = max(y)
    #y[Nrand]+=0.5*(ymax - ymin)*np.random.rand()
    #y[Nrand] =  stats.norm.rvs(size = 1, loc = (np.mean(y)), scale = ymax-ymin)
   # plt.errorbar(x, y, yerr=sigma_y, fmt=".k", capsize=0)

    #fiting
    A = np.vander(x, 2)
    C = np.diag(sigma_0 * sigma_0)
    ATA = np.dot(A.T, A / (sigma_0 ** 2)[:, None])
    cov = np.linalg.inv(ATA)
    w = np.linalg.solve(ATA, np.dot(A.T, y / (sigma_0) ** 2))
    
    a_calc.append(w[0])
    b_calc.append(w[1])
    std_a_calc.append(np.sqrt(cov[0, 0]))
    

axs[1].hist(a_calc, bins=300, histtype="stepfilled", alpha=0.3, density=True)
F_fit = np.linspace(1.0, 3.0,1000)
pdf = stats.norm(np.mean(a_calc), np.std(a_calc)).pdf(F_fit)

axs[1].plot(F_fit, pdf, '-k')
axs[1].set_xlabel("a"); plt.ylabel("P(a)")
axs[1].text(0.6,2.5, "valeur moyenne :\n a = {0:.2f} ± {1:.2f} ".format(np.mean(a_calc),np.std(a_calc)),fontsize=8)
axs[1].set_ylim(0, 3)
axs[1].set_xlim(0.5, 3.5)
axs[1].xaxis.set_tick_params(labelsize=8)
axs[1].yaxis.set_tick_params(labelsize=8)

"""

#Exemple 2 : Incertitudes non constantes suivant y
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(17)

# Choose the "true" parameters.
a_true = 2
b_true = -5


# Generate some synthetic data from the model.
Nb_test = 10000
N = 20
x = 0.5 + np.sort(9 * np.random.rand(N))
sigma_y = 0.5 + 3 * np.random.rand(N)
y = a_true * x + b_true
yerr = stats.norm.rvs(size = N, loc = 0, scale = sigma_y)
y+= yerr
x0 = np.linspace(0, 10, 500)


#fiting
A = np.vander(x, 2)
C = np.diag(sigma_y * sigma_y)
ATA = np.dot(A.T, A / (sigma_y ** 2)[:, None])
cov = np.linalg.inv(ATA)
w = np.linalg.solve(ATA, np.dot(A.T, y / (sigma_y) ** 2))

#figure
fig, axs = plt.subplots(1,2, gridspec_kw={ 'wspace': 0.35})
fig.set_size_inches(18/2.54, 7/2.54)
axs[0].errorbar(x, y, yerr=sigma_y, fmt=".k", capsize=0)
axs[0].plot(x0, a_true * x0 + b_true, color = "#2e538e", alpha=0.8, lw=3, label="Loi initiale")
axs[0].plot(x0, np.dot(np.vander(x0, 2), w), "--k", label="Ajustement linéaire")
axs[0].legend(fontsize=12)
axs[0].legend(loc = 2)
axs[0].set_xlim(0, 10)
axs[0].set_xlabel("x")
axs[0].set_ylabel("y");
axs[0].text(5.5,-7, "Estimateurs:\na = {0:.2f} ± {1:.2f} \nb = {2:.2f} ± {3:.2f}".format(w[0], np.sqrt(cov[0, 0]),w[1], np.sqrt(cov[1, 1])),fontsize=8)
axs[0].text(0.5,6, "$y=ax+b$\na = {0:.0f} \nb = {1:.0f} ".format(a_true, b_true),fontsize=8)
axs[0].set_ylim(-10, 26)
axs[0].xaxis.set_tick_params(labelsize=8)
axs[0].yaxis.set_tick_params(labelsize=8)

#Stats
a_calc = []
b_calc = []
std_a_calc=[]


for i in range(Nb_test):
    
    # Generate some synthetic data from the model.
    x = np.sort(10 * np.random.rand(N))
    sigma_y = 0.5 + 3 * np.random.rand(N)
    y = a_true * x + b_true
    yerr = stats.norm.rvs(size = N, loc = 0, scale = sigma_y)
    y+= yerr

    #fiting
    A = np.vander(x, 2)
    C = np.diag(sigma_y * sigma_y)
    ATA = np.dot(A.T, A / (sigma_y ** 2)[:, None])
    cov = np.linalg.inv(ATA)
    w = np.linalg.solve(ATA, np.dot(A.T, y / (sigma_y) ** 2))
    
    a_calc.append(w[0])
    b_calc.append(w[1])
    std_a_calc.append(np.sqrt(cov[0, 0]))
    

axs[1].hist(a_calc, bins=300, histtype="stepfilled", alpha=0.3, density=True)
F_fit = np.linspace(1.0, 3.0,1000)
pdf = stats.norm(np.mean(a_calc), np.std(a_calc)).pdf(F_fit)

axs[1].plot(F_fit, pdf, '-k')
axs[1].set_xlabel("a"); plt.ylabel("P(a)")
axs[1].text(0.6,3, "valeur moyenne :\n a = {0:.2f} ± {1:.2f} ".format(np.mean(a_calc),np.std(a_calc)),fontsize=8)
axs[1].set_ylim(0, 3.5)
axs[1].set_xlim(0.5, 3.5)
axs[1].xaxis.set_tick_params(labelsize=8)
axs[1].yaxis.set_tick_params(labelsize=8)



"""



"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(16)

# Choose the "true" parameters.
a_true = 1.084
b_true = -4.956
#sigma = 0.3


# Generate some synthetic data from the model.
N = 20
x = np.sort(10 * np.random.rand(N))
sigma_y = 0.1 + 0.4 * np.random.rand(N)
y = a_true * x + b_true

yerr = stats.norm.rvs(size = N, loc = 0, scale = sigma_y)
y+= yerr

#y += np.abs(sigma * y) * np.random.randn(N)
#y += yerr * np.random.randn(N)

# plt.errorbar(x, y, yerr=sigma_y, fmt=".k", capsize=0)
#plt.errorbar(x, y, yerr=sigma_y, fmt=".k", capsize=0)
x0 = np.linspace(0, 10, 500)
# plt.plot(x0, a_true * x0 + b_true, "k", alpha=0.3, lw=3)
# plt.xlim(0, 10)
# plt.xlabel("x")
# plt.ylabel("y");



#fiting
A = np.vander(x, 2)
C = np.diag(sigma_y * sigma_y)
ATA = np.dot(A.T, A / (sigma_y ** 2)[:, None])
cov = np.linalg.inv(ATA)
w = np.linalg.solve(ATA, np.dot(A.T, y / (sigma_y) ** 2))
print("Least-squares estimates:")
print("m = {0:.3f} ± {1:.3f}".format(w[0], np.sqrt(cov[0, 0])))
print("b = {0:.3f} ± {1:.3f}".format(w[1], np.sqrt(cov[1, 1])))

plt.errorbar(x, y, yerr=sigma_y, fmt=".k", capsize=0)
plt.plot(x0, a_true * x0 + b_true, "k", alpha=0.3, lw=3, label="Loi initiale")
plt.plot(x0, np.dot(np.vander(x0, 2), w), "--k", label="Ajustement linéaire")
plt.legend(fontsize=14)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y");
#plt.text(1,1, 'm = {0:.3f} ± {1:.3f}'.format(w[0], np.sqrt(cov[0, 0])))

plt.text(5,-3, "Estimation des coefficients :\na = {0:.3f} ± {1:.3f} \nb = {2:.3f} ± {3:.3f}".format(w[0], np.sqrt(cov[0, 0]),w[1], np.sqrt(cov[1, 1])))
plt.text(1,1, "Loi initiale : $y=ax+b$\na = {0:.3f} \nb = {1:.3f} ".format(a_true, b_true))

"""

"""
# Exemple 3 : Méthode des moindres carrés avec valeurs abérantes


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(17)

# Choose the "true" parameters.
a_true = 2
b_true = -5
sigma = 2


# Generate some synthetic data from the model.
Nb_test = 50000
N = 10
x = 0.5 + np.sort(9 * np.random.rand(N))
y = a_true * x + b_true
yerr = stats.norm.rvs(size = N, loc = 0, scale = sigma)
y+= yerr
Nrand = np.random.randint(0,N-1)
ymin = min(y)
ymax = max(y)
#y[Nrand]+=0.5*(ymax - ymin)*np.random.rand()
y[Nrand] =  stats.norm.rvs(size = 1, loc = (np.mean(y)), scale = 5*sigma)

x0 = np.linspace(0, 10, 500)


#fiting
sigma_0 = sigma * np.ones(N)
A = np.vander(x, 2)
C = np.diag(sigma_0 * sigma_0)
ATA = np.dot(A.T, A / (sigma_0 ** 2)[:, None])
cov = np.linalg.inv(ATA)
w = np.linalg.solve(ATA, np.dot(A.T, y / (sigma_0) ** 2))

#figure
fig, axs = plt.subplots(1,2, gridspec_kw={ 'wspace': 0.35})
fig.set_size_inches(18/2.54, 7/2.54)
axs[0].errorbar(x, y, yerr=sigma_0, fmt=".k", capsize=0)
axs[0].plot(x0, a_true * x0 + b_true, color = "#2e538e", alpha=0.8, lw=3, label="Loi initiale")
axs[0].plot(x0, np.dot(np.vander(x0, 2), w), "--k", label="Ajustement linéaire")
axs[0].legend(fontsize=12)
axs[0].legend(loc = 2)
axs[0].set_xlim(0, 10)
axs[0].set_xlabel("x")
axs[0].set_ylabel("y");
axs[0].text(5.5,-7, "Estimateurs:\na = {0:.2f} ± {1:.2f} \nb = {2:.2f} ± {3:.2f}".format(w[0], np.sqrt(cov[0, 0]),w[1], np.sqrt(cov[1, 1])),fontsize=8)
axs[0].text(0.5,6, "$y=ax+b$\na = {0:.0f} \nb = {1:.0f} ".format(a_true, b_true),fontsize=8)
axs[0].set_ylim(-10, 26)
axs[0].xaxis.set_tick_params(labelsize=8)
axs[0].yaxis.set_tick_params(labelsize=8)

#Stats
a_calc = []
b_calc = []
std_a_calc=[]


for i in range(Nb_test):
    
    # Generate some synthetic data from the model.
    x = np.sort(10 * np.random.rand(N))
    y = a_true * x + b_true
    yerr = stats.norm.rvs(size = N, loc = 0, scale = sigma_0)
    y+= yerr
    
    Nrand = np.random.randint(0,N-1)
    ymin = min(y)
    ymax = max(y)
    #y[Nrand]+=0.5*(ymax - ymin)*np.random.rand()
    y[Nrand] =  stats.norm.rvs(size = 1, loc = 5, scale = 5*sigma)

    #fiting
    A = np.vander(x, 2)
    C = np.diag(sigma_0 * sigma_0)
    ATA = np.dot(A.T, A / (sigma_0 ** 2)[:, None])
    cov = np.linalg.inv(ATA)
    w = np.linalg.solve(ATA, np.dot(A.T, y / (sigma_0) ** 2))
    
    a_calc.append(w[0])
    b_calc.append(w[1])
    std_a_calc.append(np.sqrt(cov[0, 0]))
    

axs[1].hist(a_calc, bins=300, histtype="stepfilled", alpha=0.3, density=True)
F_fit = np.linspace(1.0, 3.0,1000)
pdf = stats.norm(np.mean(a_calc), np.std(a_calc)).pdf(F_fit)

axs[1].plot(F_fit, pdf, '-k')
axs[1].set_xlabel("a"); plt.ylabel("P(a)")
axs[1].text(0.6,3, "valeur moyenne :\n a = {0:.2f} ± {1:.2f} ".format(np.mean(a_calc),np.std(a_calc)),fontsize=8)
axs[1].set_ylim(0, 3.5)
axs[1].set_xlim(0.5, 3.5)
axs[1].xaxis.set_tick_params(labelsize=8)
axs[1].yaxis.set_tick_params(labelsize=8)



"""


"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

# Choose the "true" parameters.
a_true = 1.084
b_true = -4.956
f_true = 0.534

# Generate some synthetic data from the model.
N = 20
x = np.sort(10 * np.random.rand(N))
yerr = 0.1 + 0.5 * np.random.rand(N)
y = a_true * x + b_true
#y += np.abs(f_true*y ) * np.random.randn(N)
y += yerr * np.random.randn(N)



plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
x0 = np.linspace(0, 10, 500)
plt.plot(x0, a_true * x0 + b_true, "k", alpha=0.3, lw=3)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y");


A = np.vander(x, 2)
C = np.diag(yerr * yerr)
ATA = np.dot(A.T, A / (yerr ** 2)[:, None])
cov = np.linalg.inv(ATA)
w = np.linalg.solve(ATA, np.dot(A.T, y / yerr ** 2))
print("Least-squares estimates:")
print("m = {0:.3f} ± {1:.3f}".format(w[0], np.sqrt(cov[0, 0])))
print("b = {0:.3f} ± {1:.3f}".format(w[1], np.sqrt(cov[1, 1])))

plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(x0, a_true * x0 + b_true, "k", alpha=0.3, lw=3, label="Loi initiale")
plt.plot(x0, np.dot(np.vander(x0, 2), w), "--k", label="Ajustement linéaire")
plt.legend(fontsize=14)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y");

plt.text(5,-5, "Estimation des coefficients :\na = {0:.3f} ± {1:.3f} \nb = {2:.3f} ± {3:.3f}".format(w[0], np.sqrt(cov[0, 0]),w[1], np.sqrt(cov[1, 1])))
plt.text(1,2, "Loi initiale : $y=ax+b$\na = {0:.3f} \nb = {1:.3f} ".format(a_true, b_true))

"""

"""
# Exemple 4 : Loi initiale non linéaire

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


np.random.seed(17)

# Choose the "true" parameters.
a_true = 2
b_true = -5
NonLinear = -0.02
sigma = 2


# Generate some synthetic data from the model.
Nb_test = 50000
N = 20
x = 0.5 + np.sort(9 * np.random.rand(N))
y = a_true * x + b_true + NonLinear*x*x
yerr = stats.norm.rvs(size = N, loc = 0, scale = sigma)
y+= yerr


x0 = np.linspace(0, 10, 500)


#fiting
sigma_0 = sigma * np.ones(N)
A = np.vander(x, 2)
C = np.diag(sigma_0 * sigma_0)
ATA = np.dot(A.T, A / (sigma_0 ** 2)[:, None])
cov = np.linalg.inv(ATA)
w = np.linalg.solve(ATA, np.dot(A.T, y / (sigma_0) ** 2))

#figure
fig, axs = plt.subplots(1,2, gridspec_kw={ 'wspace': 0.35})
fig.set_size_inches(18/2.54, 7/2.54)
axs[0].errorbar(x, y, yerr=sigma_0, fmt=".k", capsize=0)
axs[0].plot(x0, a_true * x0 + b_true + NonLinear*x0*x0, color = "#2e538e", alpha=0.8, lw=3, label="Loi initiale")
axs[0].plot(x0, np.dot(np.vander(x0, 2), w), "--k", label="Ajustement linéaire")
axs[0].legend(fontsize=12)
axs[0].legend(loc = 2)
axs[0].set_xlim(0, 10)
axs[0].set_xlabel("x")
axs[0].set_ylabel("y");
axs[0].text(5.5,-7, "Estimateurs:\na = {0:.2f} ± {1:.2f} \nb = {2:.2f} ± {3:.2f}".format(w[0], np.sqrt(cov[0, 0]),w[1], np.sqrt(cov[1, 1])),fontsize=8)
axs[0].text(0.5,5, "$y=\epsilon x^2 + ax+b$\na = {0:.0f} \nb = {1:.0f}\n$\epsilon =$ {2:.2f} ".format(a_true, b_true, NonLinear),fontsize=8)
axs[0].set_ylim(-10, 26)
axs[0].xaxis.set_tick_params(labelsize=8)
axs[0].yaxis.set_tick_params(labelsize=8)

#Stats
a_calc = []
b_calc = []
std_a_calc=[]


for i in range(Nb_test):
    
    # Generate some synthetic data from the model.
    x = np.sort(10 * np.random.rand(N))
    y = a_true * x + b_true + NonLinear*x*x
    yerr = stats.norm.rvs(size = N, loc = 0, scale = sigma_0)
    y+= yerr
    

    #fiting
    A = np.vander(x, 2)
    C = np.diag(sigma_0 * sigma_0)
    ATA = np.dot(A.T, A / (sigma_0 ** 2)[:, None])
    cov = np.linalg.inv(ATA)
    w = np.linalg.solve(ATA, np.dot(A.T, y / (sigma_0) ** 2))
    
    a_calc.append(w[0])
    b_calc.append(w[1])
    std_a_calc.append(np.sqrt(cov[0, 0]))
    

axs[1].hist(a_calc, bins=300, histtype="stepfilled", alpha=0.3, density=True)
F_fit = np.linspace(1.0, 3.0,1000)
pdf = stats.norm(np.mean(a_calc), np.std(a_calc)).pdf(F_fit)

axs[1].plot(F_fit, pdf, '-k')
axs[1].set_xlabel("a"); plt.ylabel("P(a)")
axs[1].text(0.6,2.5, "valeur moyenne :\n a = {0:.2f} ± {1:.2f} ".format(np.mean(a_calc),np.std(a_calc)),fontsize=8)
axs[1].set_ylim(0, 3)
axs[1].set_xlim(0.5, 3)
axs[1].xaxis.set_tick_params(labelsize=8)
axs[1].yaxis.set_tick_params(labelsize=8)


"""

"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(16)

# Choose the "true" parameters.
a1_true = 1.084
a2_true = 0.01
a0_true = -4.956
f_true = 0.534

# Generate some synthetic data from the model.
N = 20
x = np.sort(10 * np.random.rand(N))
yerr = 0.1 + 0.5 * np.random.rand(N)
y =a2_true * x*x+ a1_true * x + a0_true
#y += np.abs(f_true*y ) * np.random.randn(N)
y += yerr * np.random.randn(N)



plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
x0 = np.linspace(0, 10, 500)
plt.plot(x0,a2_true * x0* x0 + a1_true * x0 + a0_true, "k", alpha=0.3, lw=3)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y");


A = np.vander(x, 2)
C = np.diag(yerr * yerr)
ATA = np.dot(A.T, A / (yerr ** 2)[:, None])
cov = np.linalg.inv(ATA)
w = np.linalg.solve(ATA, np.dot(A.T, y / yerr ** 2))
print("Least-squares estimates:")
print("m = {0:.3f} ± {1:.3f}".format(w[0], np.sqrt(cov[0, 0])))
print("b = {0:.3f} ± {1:.3f}".format(w[1], np.sqrt(cov[1, 1])))

plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(x0, a2_true * x0* x0 + a1_true * x0 + a0_true, "k", alpha=0.3, lw=3, label="Loi initiale")
plt.plot(x0, np.dot(np.vander(x0, 2), w), "--k", label="Ajustement linéaire")
plt.legend(fontsize=14)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y");

plt.text(5,-5, "Estimation des coefficients :\n$a_1 =$ {0:.3f} ± {1:.3f} \n$a_0 =$ {2:.3f} ± {3:.3f}".format(w[0], np.sqrt(cov[0, 0]),w[1], np.sqrt(cov[1, 1])))
plt.text(1,0, "Loi initiale : $y=a_2 x^2 + a_1 x+ a_0$\n$a_0 =$ {0:.3f} \n$a_1 =$ {1:.3f}\n$a_2 =$ {2:.3f} ".format(a0_true, a1_true, a2_true))

"""


#Exemple 5 : Incertitudes non constantes suivant x

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(16)

# Choose the "true" parameters.
a_true = 10.084
b_true = -4.956
#sigma = 0.3


# Generate some synthetic data from the model.
N = 20
x = np.sort(10 * np.random.rand(N))
sigma_x = 0.1 + 0.4 * np.random.rand(N)
y = a_true * x + b_true

xerr = stats.norm.rvs(size = N, loc = 0, scale = sigma_x)
x+= xerr

#y += np.abs(sigma * y) * np.random.randn(N)
#y += yerr * np.random.randn(N)

# plt.errorbar(x, y, yerr=sigma_y, fmt=".k", capsize=0)
#plt.errorbar(x, y, yerr=sigma_y, fmt=".k", capsize=0)
x0 = np.linspace(0, 10, 500)
# plt.plot(x0, a_true * x0 + b_true, "k", alpha=0.3, lw=3)
# plt.xlim(0, 10)
# plt.xlabel("x")
# plt.ylabel("y");


sigma_y=1* np.ones(N)

#fiting
A = np.vander(x, 2)
C = np.diag(sigma_y * sigma_y)
ATA = np.dot(A.T, A / (sigma_y ** 2)[:, None])
cov = np.linalg.inv(ATA)
w = np.linalg.solve(ATA, np.dot(A.T, y / (sigma_y) ** 2))
print("Least-squares estimates:")
print("m = {0:.3f} ± {1:.3f}".format(w[0], np.sqrt(cov[0, 0])))
print("b = {0:.3f} ± {1:.3f}".format(w[1], np.sqrt(cov[1, 1])))

plt.errorbar(x, y, xerr=sigma_x, fmt=".k", capsize=0)
plt.plot(x0, a_true * x0 + b_true, "k", alpha=0.3, lw=3, label="Loi initiale")
plt.plot(x0, np.dot(np.vander(x0, 2), w), "--k", label="Ajustement linéaire")
plt.legend(fontsize=14)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y");
#plt.text(1,1, 'm = {0:.3f} ± {1:.3f}'.format(w[0], np.sqrt(cov[0, 0])))

plt.text(5,-3, "Estimation des coefficients :\na = {0:.3f} ± {1:.3f} \nb = {2:.3f} ± {3:.3f}".format(w[0], np.sqrt(cov[0, 0]),w[1], np.sqrt(cov[1, 1])))
plt.text(1,40, "Loi initiale : $y=ax+b$\na = {0:.3f} \nb = {1:.3f} ".format(a_true, b_true))

"""
"""
#Exemple 5 : Incertitudes non constantes suivant x et y


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


np.random.seed(17)

# Choose the "true" parameters.
a_true = 2
b_true = -5
sigmay = 2
sigmax = 1


# Generate some synthetic data from the model.
Nb_test = 50000
N = 20
x = 0.5 + np.sort(9 * np.random.rand(N))
y = a_true * x + b_true 
yerr = stats.norm.rvs(size = N, loc = 0, scale = sigmay)
xerr = stats.norm.rvs(size = N, loc = 0, scale = sigmax)
y += yerr
x += xerr

x0 = np.linspace(0, 10, 500)


#fiting
sigma_0 = sigmay * np.ones(N)
A = np.vander(x, 2)
C = np.diag(sigma_0 * sigma_0)
ATA = np.dot(A.T, A / (sigma_0 ** 2)[:, None])
cov = np.linalg.inv(ATA)
w = np.linalg.solve(ATA, np.dot(A.T, y / (sigma_0) ** 2))

#figure
fig, axs = plt.subplots(1,2, gridspec_kw={ 'wspace': 0.35})
fig.set_size_inches(18/2.54, 7/2.54)
axs[0].errorbar(x, y, yerr=sigmay, xerr=sigmax, fmt=".k", capsize=0)
axs[0].plot(x0, a_true * x0 + b_true , color = "#2e538e", alpha=0.8, lw=3, label="Loi initiale")
axs[0].plot(x0, np.dot(np.vander(x0, 2), w), "--k", label="Ajustement linéaire")
axs[0].legend(fontsize=12)
axs[0].legend(loc = 2)
axs[0].set_xlim(0, 10)
axs[0].set_xlabel("x")
axs[0].set_ylabel("y");
axs[0].text(5.5,-7, "Estimateurs:\na = {0:.2f} ± {1:.2f} \nb = {2:.2f} ± {3:.2f}".format(w[0], np.sqrt(cov[0, 0]),w[1], np.sqrt(cov[1, 1])),fontsize=8)
axs[0].text(0.5,5, "$y=ax+b$\na = {0:.0f} \nb = {1:.0f} ".format(a_true, b_true),fontsize=8)
axs[0].set_ylim(-10, 26)
axs[0].xaxis.set_tick_params(labelsize=8)
axs[0].yaxis.set_tick_params(labelsize=8)

#Stats
a_calc = []
b_calc = []
std_a_calc=[]


for i in range(Nb_test):
    
    # Generate some synthetic data from the model.
    x = np.sort(10 * np.random.rand(N))
    y = a_true * x + b_true 
    yerr = stats.norm.rvs(size = N, loc = 0, scale = sigmay)
    y+= yerr
    xerr = stats.norm.rvs(size = N, loc = 0, scale = sigmax)
    x += xerr
    

    #fiting
    A = np.vander(x, 2)
    C = np.diag(sigma_0 * sigma_0)
    ATA = np.dot(A.T, A / (sigma_0 ** 2)[:, None])
    cov = np.linalg.inv(ATA)
    w = np.linalg.solve(ATA, np.dot(A.T, y / (sigma_0) ** 2))
    
    a_calc.append(w[0])
    b_calc.append(w[1])
    std_a_calc.append(np.sqrt(cov[0, 0]))
    

axs[1].hist(a_calc, bins=300, histtype="stepfilled", alpha=0.3, density=True)
F_fit = np.linspace(1.0, 3.0,1000)
pdf = stats.norm(np.mean(a_calc), np.std(a_calc)).pdf(F_fit)

axs[1].plot(F_fit, pdf, '-k')
axs[1].set_xlabel("a"); plt.ylabel("P(a)")
axs[1].text(0.6,2.5, "valeur moyenne :\n a = {0:.2f} ± {1:.2f} ".format(np.mean(a_calc),np.std(a_calc)),fontsize=8)
axs[1].set_ylim(0, 3)
axs[1].set_xlim(0.5, 3)
axs[1].xaxis.set_tick_params(labelsize=8)
axs[1].yaxis.set_tick_params(labelsize=8)



"""

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(123)

# Choose the "true" parameters.
a_true = 1.084
b_true = -4.956
#sigma = 0.3


# Generate some synthetic data from the model.
N = 20
x = np.sort(10 * np.random.rand(N))
sigma_x = 0.1 + 0.4 * np.random.rand(N)
sigma_y = 0.1 + 0.4 * np.random.rand(N)
#y = 0.1 + 0.4 * np.random.rand(N)
y = a_true * x + b_true

xerr = stats.norm.rvs(size = N, loc = 0, scale = sigma_x)
x+= xerr

yerr = stats.norm.rvs(size = N, loc = 0, scale = sigma_y)
y+= yerr

#y += np.abs(sigma_y * y) * np.random.randn(N)
#y += yerr * np.random.randn(N)

# plt.errorbar(x, y, yerr=sigma_y, fmt=".k", capsize=0)
#plt.errorbar(x, y, yerr=sigma_y, fmt=".k", capsize=0)
x0 = np.linspace(0, 10, 500)
# plt.plot(x0, a_true * x0 + b_true, "k", alpha=0.3, lw=3)
# plt.xlim(0, 10)
# plt.xlabel("x")
# plt.ylabel("y");



#fiting
A = np.vander(x, 2)
C = np.diag(sigma_y * sigma_y)
ATA = np.dot(A.T, A / (sigma_y ** 2)[:, None])
cov = np.linalg.inv(ATA)
w = np.linalg.solve(ATA, np.dot(A.T, y / (sigma_y) ** 2))
print("Least-squares estimates:")
print("m = {0:.3f} ± {1:.3f}".format(w[0], np.sqrt(cov[0, 0])))
print("b = {0:.3f} ± {1:.3f}".format(w[1], np.sqrt(cov[1, 1])))

plt.errorbar(x, y,yerr = sigma_y, xerr=sigma_x, fmt=".k", capsize=0)
plt.plot(x0, a_true * x0 + b_true, "k", alpha=0.3, lw=3, label="Loi initiale")
plt.plot(x0, np.dot(np.vander(x0, 2), w), "--k", label="Ajustement linéaire")
plt.legend(fontsize=14)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y");
#plt.text(1,1, 'm = {0:.3f} ± {1:.3f}'.format(w[0], np.sqrt(cov[0, 0])))

plt.text(5,-3, "Estimation des coefficients :\na = {0:.3f} ± {1:.3f} \nb = {2:.3f} ± {3:.3f}".format(w[0], np.sqrt(cov[0, 0]),w[1], np.sqrt(cov[1, 1])))
plt.text(1,1, "Loi initiale : $y=ax+b$\na = {0:.3f} \nb = {1:.3f} ".format(a_true, b_true))

"""
"""
# Etude dispersion pente et ordonnée origine

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(128)

# Choose the "true" parameters.
a_true = 2
b_true = -5


#Stats
Nb_test = 100000
a_calc = []
b_calc = []
std_a_calc=[]


for i in tqdm(range(Nb_test)):
    
    # Generate some synthetic data from the model.
    N = 20
    x = np.sort(10 * np.random.rand(N))
    sigma_y = 0.1 + 0.4 * np.random.rand(N)
    y = a_true * x + b_true
    yerr = stats.norm.rvs(size = N, loc = 0, scale = sigma_y)
    y+= yerr
    Nrand = np.random.randint(0,N-1)
    ymin = min(y)
    ymax = max(y)
    #y[Nrand]+=0.5*(ymax - ymin)*np.random.rand()
    y[Nrand] =  stats.norm.rvs(size = 1, loc = (np.mean(y)), scale = ymax-ymin)
   # plt.errorbar(x, y, yerr=sigma_y, fmt=".k", capsize=0)

    #fiting
    A = np.vander(x, 2)
    C = np.diag(sigma_y * sigma_y)
    ATA = np.dot(A.T, A / (sigma_y ** 2)[:, None])
    cov = np.linalg.inv(ATA)
    w = np.linalg.solve(ATA, np.dot(A.T, y / (sigma_y) ** 2))
    
    a_calc.append(w[0])
    b_calc.append(w[1])
    std_a_calc.append(np.sqrt(cov[0, 0]))
    

plt.hist(a_calc, bins=200, histtype="stepfilled", alpha=0.3, density=True)
F_fit = np.linspace(1.5, 2.5,1000)
pdf = stats.norm(np.mean(a_calc), np.std(a_calc)).pdf(F_fit)

plt.plot(F_fit, pdf, '-k')
plt.xlabel("a"); plt.ylabel("P(a)")
plt.text(np.mean(a_calc)+2*np.std(a_calc),1.2, "valeur moyenne :\n a = {0:.3f} ± {1:.3f} ".format(np.mean(a_calc),np.std(a_calc)))



print("Least-squares estimates:")
print("mean of standard deviation of a : {0:.3f}".format(np.mean(std_a_calc)))




"""

"""
# Changement de variable



import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

np.random.seed(17)

# Choose the "true" parameters.
a_true = 2
b_true = 0
sigmay = 0.3

CstG = 9.81
a_true = 4*(np.pi**2)/CstG


# Generate some synthetic data from the model.
Nb_test = 10000
N = 20
#x = 0.1 + np.sort(2 * np.random.rand(N))
x = np.linspace(0.1, 2, N)
y1 =2*np.pi*np.sqrt( x /CstG )
yerr = stats.norm.rvs(size = N, loc = 0, scale = sigmay)
y1 += yerr

y = np.square(y1) 


x0 = np.linspace(0.1, 2, 500)


#fiting
sigma_0 = 2*y*sigmay * np.ones(N)
A = np.vander(x, 2)
C = np.diag(sigma_0 * sigma_0)
ATA = np.dot(A.T, A / (sigma_0 ** 2)[:, None])
cov = np.linalg.inv(ATA)
w = np.linalg.solve(ATA, np.dot(A.T, y / (sigma_0) ** 2))

#figure
fig, axs = plt.subplots(1,2, gridspec_kw={ 'wspace': 0.35})
fig.set_size_inches(18/2.54, 7/2.54)
axs[0].errorbar(x, y, yerr=abs(2*y*sigmay), fmt=".k", capsize=0)
axs[0].plot(x0, 4*(np.pi**2)/CstG * x0  , color = "#2e538e", alpha=0.8, lw=3, label="Loi initiale")
axs[0].plot(x0, np.dot(np.vander(x0, 2), w), "--k", label="Ajustement lineaire")
axs[0].legend(fontsize=12)
axs[0].legend(loc = 2)
axs[0].set_xlim(0, 2.1)
axs[0].set_xlabel("x")
axs[0].set_ylabel("y");
axs[0].text(1,15, "Estimateurs:\na = {0:.2f} ± {1:.2f} \nb = {2:.2f} ± {3:.2f}".format(w[0], np.sqrt(cov[0, 0]),w[1], np.sqrt(cov[1, 1])),fontsize=8)
axs[0].text(0.1,15, "$y=ax+b$\na = {0:.2f} \nb = {1:.0f} ".format(a_true, b_true),fontsize=8)
axs[0].set_ylim(-3, 31)
axs[0].xaxis.set_tick_params(labelsize=8)
axs[0].yaxis.set_tick_params(labelsize=8)

#Stats
a_calc = []
b_calc = []
std_a_calc=[]


for i in tqdm(range(Nb_test)):
    
    # Generate some synthetic data from the model.
    #x = 0.1 + np.sort(2 * np.random.rand(N))
    np.linspace(0, 2, 500)
    y1 =2*np.pi*np.sqrt( x/CstG )
    yerr = stats.norm.rvs(size = N, loc = 0, scale = sigmay)
    y1 += yerr

    y = np.square(y1)  
    

    #fiting
    sigma_0 = 2*y*sigmay * np.ones(N)
    A = np.vander(x, 2)
    C = np.diag(sigma_0 * sigma_0)
    ATA = np.dot(A.T, A / (sigma_0 ** 2)[:, None])
    cov = np.linalg.inv(ATA)
    w = np.linalg.solve(ATA, np.dot(A.T, y / (sigma_0) ** 2))
    
    a_calc.append(w[0])
    b_calc.append(w[1])
    std_a_calc.append(np.sqrt(cov[0, 0]))
    

g_values =(4*np.pi**2)* np.power(a_calc, -1)

axs[1].hist(g_values, bins=300, histtype="stepfilled", alpha=0.3, density=True)



F_fit = np.linspace(7, 14,1000)
pdf = stats.norm(np.mean(g_values), np.std(g_values)).pdf(F_fit)

axs[1].plot(F_fit, pdf, '-k')
axs[1].set_xlabel("g"); plt.ylabel("P(g)")
axs[1].text(7.5,0.7, "valeur theorique :\n g = 9.81\nvaleur estimee\n g={2:.2f} ± {3:.2f} ".format(np.mean(a_calc),np.std(a_calc),np.mean(g_values),np.std(g_values)),fontsize=8)
axs[1].set_ylim(0, 1)
axs[1].set_xlim(7, 14)
axs[1].xaxis.set_tick_params(labelsize=8)
axs[1].yaxis.set_tick_params(labelsize=8)

print(np.mean(b_calc))

plt.savefig("C:/Users/Romain/Desktop/Mesures/Images/ChangementDeVariable.pdf", bbox_inches="tight")
"""

# sigma corrélé


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

np.random.seed(23)

# Choose the "true" parameters.
a_true = 2
b_true = -5
sigma = 0.15

# Generate some synthetic data from the model.
Nb_test = 5000
N = 20
x = np.linspace(0.1, 20, N)# 0.5 + np.sort(9 * np.random.rand(N))
#yerr = 0.2 #+ 0.2 * np.random.rand(N)
y = a_true * x + b_true
yerr = stats.norm.rvs(size = N, loc = 0, scale = abs(y*sigma))
y+= yerr
x0 = np.linspace(0, 20, 500)


#fiting
sigma_0 = abs(y* sigma) * np.ones(N)
A = np.vander(x, 2)
C = np.diag(sigma_0 * sigma_0)
ATA = np.dot(A.T, A / (sigma_0 ** 2)[:, None])
cov = np.linalg.inv(ATA)
w = np.linalg.solve(ATA, np.dot(A.T, y / (sigma_0) ** 2))

#figure
fig, axs = plt.subplots(1,2, gridspec_kw={ 'wspace': 0.35})
fig.set_size_inches(18/2.54, 7/2.54)
axs[0].errorbar(x, y, yerr=abs(y*sigma), fmt=".k", capsize=0)
axs[0].plot(x0, a_true * x0 + b_true, color = "#2e538e", alpha=0.8, lw=3, label="Loi initiale")
axs[0].plot(x0, np.dot(np.vander(x0, 2), w), "--k", label="Ajustement linéaire")
axs[0].legend(fontsize=12)
axs[0].legend(loc = 2)
axs[0].set_xlim(0, 20)
axs[0].set_xlabel("x")
axs[0].set_ylabel("y");
axs[0].text(5.5,-7, "Estimateurs:\na = {0:.2f} ± {1:.2f} \nb = {2:.2f} ± {3:.2f}".format(w[0], np.sqrt(cov[0, 0]),w[1], np.sqrt(cov[1, 1])),fontsize=8)
axs[0].text(0.5,6, "$y=ax+b$\na = {0:.0f} \nb = {1:.0f} ".format(a_true, b_true),fontsize=8)
axs[0].set_ylim(-10, 46)
axs[0].xaxis.set_tick_params(labelsize=8)
axs[0].yaxis.set_tick_params(labelsize=8)

#Stats
a_calc = []
b_calc = []
std_a_calc=[]


for i in tqdm(range(Nb_test)):
    
    # Generate some synthetic data from the model.
    #x = np.sort(10 * np.random.rand(N))
    x = np.linspace(0.1, 10, N)
    sigma_y = 0.1 + 0.4 * np.random.rand(N)
    y = a_true * x + b_true
    yerr = stats.norm.rvs(size = N, loc = 0, scale = abs(y*sigma))
    y+= yerr

    #fiting
    sigma_0 = abs(y* sigma) * np.ones(N)
    A = np.vander(x, 2)
    C = np.diag(sigma_0 * sigma_0)
    ATA = np.dot(A.T, A / (sigma_0 ** 2)[:, None])
    cov = np.linalg.inv(ATA)
    w = np.linalg.solve(ATA, np.dot(A.T, y / (sigma_0) ** 2))
    
    a_calc.append(w[0])
    b_calc.append(w[1])
    std_a_calc.append(np.sqrt(cov[0, 0]))
    

axs[1].hist(a_calc, bins=300, histtype="stepfilled", alpha=0.3, density=True)
F_fit = np.linspace(1.0, 3.0,1000)
pdf = stats.norm(np.mean(a_calc), np.std(a_calc)).pdf(F_fit)

axs[1].plot(F_fit, pdf, '-k')
axs[1].set_xlabel("a"); plt.ylabel("P(a)")
axs[1].text(1.52,5.4, "valeur moyenne :\n a = {0:.2f} ± {1:.2f} ".format(np.mean(a_calc),np.std(a_calc)),fontsize=8)
axs[1].set_ylim(0, 6)
axs[1].set_xlim(1.5, 2.5)
axs[1].xaxis.set_tick_params(labelsize=8)
axs[1].yaxis.set_tick_params(labelsize=8)

#plt.savefig("C:/Users/Romain/Desktop/Mesures/Images/LinearFitCorrelatedUncertainty.pdf", bbox_inches="tight")