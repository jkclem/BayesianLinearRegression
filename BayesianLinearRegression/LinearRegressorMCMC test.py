# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 05:09:27 2020

@author: jkcle
"""
import matplotlib.pyplot as plt
import numpy as np
from LinearRegressorMCMC import LinearRegressorMCMC

X = np.arange(1, 101, 1)
y = 0.5 + 0.75*X + np.random.normal(size=X.shape[0])
X = X.reshape((-1,1))
y = y.reshape((-1, 1))
lrmcmc = LinearRegressorMCMC()
lrmcmc.fit(y,
            1,        
            X, 
            np.repeat(0.6, 2),
            np.repeat(1, 2),
            np.repeat(0.01, 2), 
            n_iter=20000,
            add_intercept=True,
            method="median",
            burn_in=0.1,
            alpha=0.05,
            random_seed=1)

plt.plot(lrmcmc._beta_distribution[0], lrmcmc._beta_distribution[1])