# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 00:48:35 2020

@author: jkcle
"""
import numpy as np
import pandas as pd
from LogisticRegressorMCMC import LogisticRegressorMCMC
import matplotlib.pyplot as plt


# load in field goal data
all_data = pd.read_csv('candy-data.csv')

# list of independent variables in the model
vars_of_interest = ['pricepercent']
# name of dependent variable
target = 'chocolate'
my_data = all_data[[target] + vars_of_interest]
my_data = all_data.dropna()


# make a column vector numpy array for the dependent variable
y = np.array(my_data[target]).reshape((-1,1))
# make a column vector numpy array for the independent variable
X = np.array(my_data[vars_of_interest]).reshape((-1,len(vars_of_interest)))

# make a numpy array for the beta means priors
beta_priors = np.repeat(0.0, len(vars_of_interest)+1) 
# make a numpy array for the beta standard deviation priors
prior_stds = np.repeat(1, len(vars_of_interest)+1)
# make a row vector numpy array for the standard deviation of the jumper distribution
jumper_stds = np.repeat(0.1, len(vars_of_interest)+1) 
# set the number of iterations
n_iter = 50000

lrmcmc = LogisticRegressorMCMC()
lrmcmc.fit(y, X, beta_priors, prior_stds, jumper_stds, n_iter, method='mode', 
           add_intercept=True)

plt.plot(lrmcmc._beta_distribution[0], lrmcmc._beta_distribution[1])