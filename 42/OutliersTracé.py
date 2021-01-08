# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 13:07:02 2020

@author: Romain
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 09:16:11 2020

@author: Romain
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:30:40 2020

@author: Romain
"""




# tracé graphique
xmax = 20
ymax = 20
fig, ax = plt.subplots(figsize=(xmax/2.54, ymax/2.54))

ax.errorbar(x, y, yerr=abs(sigma), fmt='.k', ecolor='gray', elinewidth = 1, label = 'Données')
ax.errorbar(x[Nrand], y[Nrand], yerr=abs(sigma[Nrand]), fmt='.r', ecolor='red', elinewidth = 1, label = 'Données aberrantes')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim(-4,104)
ax.legend(fontsize=11)




ax.plot(x, a_true * x + b_true, "b", alpha=0.4, lw=3, label="modéle initial")
ax.plot(x, np.dot(np.vander(x, 2), w), "--k", label="Régression linéaire")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(fontsize=11)



#ax.plot(x, theta3[0] + theta3[1] * x, color='black', label = "Ajustement par inférence bayésienne")
ax.plot(x[outliers], y[outliers], 'ro', ms=14, mfc='none', mec='blue', label='Données estimées comme aberrantes')
plt.title('Ajustement par regression : inférence bayésienne')

plot_MCMC_results(x, y, emcee_trace)
ax.legend(fontsize=11)
ax.set_ylim(-30, 250)

fig.savefig("C:/Users/Romain/Desktop/Mesures/Images/AjustementParInferenceBayesienne.pdf", bbox_inches='tight', transparent=True, pad_inches=0.5)


#import corner

"""
flat_samples = sampler.get_chain(discard=nburn, thin=15, flat=True)

inds = np.random.randint(len(flat_samples), size=100)
for ind in inds:
    sample = flat_samples[ind]
    ax.plot(x, sample[0] + sample[1]*x, "C1", alpha=0.1, zorder = -1)
"""

