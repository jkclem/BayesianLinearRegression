# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 16:51:38 2020

@author: jkcle
"""
from RegressorMCMC import RegressorMCMC
import numpy as np

class LogisticRegressorMCMC(RegressorMCMC):
    
    def __init__(self):
        RegressorMCMC.__init__(self)
        
    def inv_logit(self, beta, X):
        """
        This method takes in a vector of coefficients for a logistic 
        regression model and a matrix of data and returns the probabilities of
        belonging to the class 1 by first calculating the log-odds and 
        translating the log-odds to probabilities. It is used by the 
        log_likelihood and predict_probabilities methods.

        Parameters
        ----------
        beta : numpy array
            A 1-D vector of coefficients in a logistic regression model.
        X : numpy array
            A 2-D matrix where rows represent observations and columns 
            represent variables.

        Returns
        -------
        probabilities : numpy array
            A 1-D vector of the probabilities associated from the logistic
            regression.

        """
        
        # Calculate the numerator of the inverse logit transformation.
        numerator = np.exp(np.matmul(X, beta))
        # Calculate the denominator of the inverse logit transformation.
        denominator = 1 + np.exp(np.matmul(X, beta))
        # Calculate the probabilities.
        probabilities =  numerator / denominator 
        
        return probabilities
    
    def log_likelihood(self, y, X, beta):
        """
        Overwrites the log_likelihood method inherited from the RegressorMCMC
        class to calculate the log-likelihood of the logistic regression
        coefficients given binomially-distributed data. It is used in the
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
        log_likelihood : float
            The log-likelihood of the beta vector given the data.

        """
        
        # Calculate the log-likelihood of beta given the data.
        log_likelihood = np.sum(y*np.log(self.inv_logit(beta, X)) 
                                + (1-y)*np.log((1-self.inv_logit(beta,X))))
        
        return log_likelihood
    
    def predict_probabilities(self, X):
        """
        This method returns predictions of belonging to class 1 in 
        probabilities because the predict method will give predictions in 
        log-odds.

        Parameters
        ----------
        X : numpy array
            A 2-D matrix where rows represent observations and columns 
            represent variables.

        Returns
        -------
        predicted_probabilities : numpy array
            A 1-D array of the predicted probabilites of belonging to class 1.

        """
        
        # Add a column of ones in the first column if this instance of the
        # class is fit with an intercept.
        if self._intercept_added:
            X_new = np.append(np.ones(shape=(X.shape[0], 1)), X, 1)
        # Do not add a column of ones if it was not added by this instance.
        else:
            X_new = X
        
        # Calculate the probability of each new observation belonging to 
        # class 1.
        predicted_probabilities = self.inv_logit(self._beta_hat, X_new)
            
        return predicted_probabilities
    
    def predict_classes(self, X, boundary=0.5):
        """
        This method predicts the class of new observations based on a decision 
        boundary for probability. If predicted probability >= boundary, it is
        predicted to belong to class 1.

        Parameters
        ----------
        X : numpy array
            A 2-D matrix where rows represent observations and columns 
            represent variables.
        boundary : float, optional
            A float on the closed interval between 0 and 1 and is the minimum
            predicted probability to classify a new observation of belonging 
            to class 1. The default is 0.5.

        Returns
        -------
        predicted_classes : numpy array
            DESCRIPTION.

        """

        # Predict the probabilities of belonging to class 1.
        predicted_probabilities = self.predict_probabilities(X)
        # Set predictions to 1 or 0 based on the decision boundary.
        predicted_classes = np.where(predicted_probabilities >= boundary, 1, 0)
        
        return predicted_classes