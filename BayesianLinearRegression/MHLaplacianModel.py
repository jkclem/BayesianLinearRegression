# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 08:10:34 2020

@author: jkcle
"""
from .MHLinearModel import MetropolisHastingsLinearModel
from scipy.stats import laplace
import numpy as np

class LaplacianModel(MetropolisHastingsLinearModel):
    """
    The LaplacianModel class inherits from the MetropolisHastingsLinearModel
    class. The _log_likelihood method for laplace distributed data is defined
    here. The LaplacianModel class must be initialized with a value for the
    standard deviation of y given X. An estimate can be obtained from the 
    output of a Least Absolute Deviation regression.
    """
    
    def __init__(self, y_scale):
        MetropolisHastingsLinearModel.__init__(self)
        self._y_scale = y_scale
        
    def _log_likelihood(self, y, X, beta):
        """
        Overwrites the _log_likelihood method inherited from the RegressorMCMC
        class to calculate the log-likelihood of the linear regression
        coefficients given normally-distributed data. It is used in the
        model fitting process. 

        Parameters
        ----------
        y : numpy array
            A 1-D vector of 0s and 1s representing the two classes.
        X : numpy array
            A 2-D matrix where rows represent observations and columns 
            represent variables.
        beta : numpy array
            A 1-D vector of coefficients in a logistic regression model.

        Returns
        -------
        _log_likelihood : float
            The log-likelihood of the beta vector given the data.

        """
        
        # Predict y given the current coefficients and X.
        predicted_y = np.matmul(X, beta)
        
        # Calculate the log-likelihood of beta given the data.
        _log_likelihood = np.sum(laplace.logpdf(y,
                                                loc=predicted_y,
                                                scale=self._y_scale))
        
        return _log_likelihood