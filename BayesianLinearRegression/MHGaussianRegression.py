# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 04:22:39 2020

@author: jkcle
"""
from .MHLinearModel import MHLinearRegressor
from scipy.stats import norm
import numpy as np
from tqdm import tqdm

class GaussianRegressor(MHLinearRegressor):
    
    def __init__(self):
        MHLinearRegressor.__init__(self)
        
    def _log_likelihood(self, y, X, beta, y_std):
        """
        Overwrites the _log_likelihood method inherited from the RegressorMCMC
        class to calculate the log-likelihood of the linear regression
        coefficients given normally-distributed data. It is used in the
        model fitting process. y_std must be estimated prior to using this
        function. Using the estimate from an OLS model is recommended.

        Parameters
        ----------
        y : numpy array
            A 1-D vector of 0s and 1s representing the two classes.
        X : numpy array
            A 2-D matrix where rows represent observations and columns 
            represent variables.
        beta : numpy array
            A 1-D vector of coefficients in a logistic regression model.
        y_std : float
            The estimated standard deviation of y. OLS can provide a good 
            estimate.

        Returns
        -------
        _log_likelihood : float
            The log-likelihood of the beta vector given the data.

        """
        
        # Predict y given the current coefficients and X.
        predicted_y = np.matmul(X, beta)
        
        # Calculate the log-likelihood of beta given the data.
        _log_likelihood = np.sum(norm.logpdf(y,
                                            loc=predicted_y,
                                            scale=y_std))
        
        return _log_likelihood
    
    def _log_posterior(self, y, X, beta, prior_means, prior_stds, y_std):
        """
        This method calculates a value proportional to the log-posterior of 
        the betas given the log-likelihood of the proposed coefficients given
        the data and the log-prior. It is used in the model fitting process.

        Parameters
        ----------
        y : numpy array
            A 1-D array of the endogenous variable (target variable).
        X : numpy array
            A 2-D matrix where rows represent observations and columns 
            represent variables.
        beta : numpy array
            A 1-D vector of coefficients in the regression model.
        prior_means : numpy array
            A 1-D vector of means for the prior distributions of the 
            coefficients.
        prior_stds : numpy array
            A 1-D vector of standard devations for the prior distributions of
            the coefficients.
         y_std : float
            The estimated standard deviation of y. OLS can provide a good 
            estimate.

        Returns
        -------
        _log_posterior : float
            A value proportional to the log-posterior.

        """   
        
        # Calculate a value proportional to the log-posterior.
        _log_posterior = (self.normal_log_prior(beta, prior_means, prior_stds) 
                         + self._log_likelihood(y, X, beta, y_std))
        
        return _log_posterior
    
    
    def fit(self, 
            y,
            y_std,        
            X, 
            prior_means,
            prior_stds,
            jumper_stds, 
            n_iter=10000,
            add_intercept=True,
            method="mediean",
            burn_in=0.1,
            alpha=0.05,
            random_seed=1):
        """
        This method is copied from the parent class, but modified because the
        _log_likelihood method has an additional input. 
        
        This method fits a model to the input data by simulating the posterior
        distributions of the coefficients using the Metropolis-Hastings 
        algorithm.

        Parameters
        ----------
        y : numpy array
            A 1-D array of the endogenous variable (target variable).
        y_std : float
            The estimated standard deviation of y. OLS can provide a good 
            estimate.
        X : numpy array
            A 2-D matrix where rows represent observations and columns 
            represent variables.
        prior_means : numpy array
            A 1-D vector of means for the prior distributions of the 
            coefficients.
        prior_stds : numpy array
            A 1-D vector of standard devations for the prior distributions of
            the coefficients.
        jumper_stds : numpy array
            A 1-D vector of standard devations for the jumping distribution 
            used to propose new values for the coefficients.
        n_iter : int, optional
            The number of iterations used in the Metropolis-Hastings algorithm. 
            The default is 10000.
        add_intercept : bool, optional
            If True, a column of ones is inserted in the first column of X and
            is for calculating an intercept. The default is True.
        method : str, optional
            Determines whether the _beta_hat attribute uses either the median,
            mean, or mode of the _beta_distribution attribute as the estimates
            of the regression coefficients. The default is "mediean".
        burn_in : float, optional
            Discards the first burn_in % of samples. The default is 0.1.
        alpha : float, optional
            Determines the credible intervals widths. The default is 0.05.
        random_seed : int, optional
            Sets a random seed for reproducibility. The default is 1.

        Returns
        -------
        None.

        """
        
        # Set a random seed.
        np.random.seed(random_seed)
        
        # Add an intercept if desired by inserting a column of ones in the
        # first column place. Otherwise, copy the data.
        if add_intercept:
            X_mod = np.append(np.ones(shape=(X.shape[0], 1)),X, 1)
        else:
            X_mod = X
        
        # Delete X from memory inside the function to free memory.
        del X
        
        assert (prior_means.shape[0]
                == prior_stds.shape[0]
                == jumper_stds.shape[0]
                == X_mod.shape[1]), ("The prior and jumping distribution "
                                     "parameters must match the number of "
                                     "coefficients.")
        
            
        # Create a list of beta indexes to be looped through each iteration.
        beta_indexes = [k for k in range(len(prior_means))]
        
        # Initialize beta_hat with the priors. It will have a number of rows
        # equal to the number of beta coefficients and a number of columns
        # equal to the number of iterations, + 1 for the prior. Each row will 
        # hold values of a single coefficient. Each column is an iteration
        # of the algorithm.
        beta_hat = np.array(np.repeat(prior_means, n_iter + 1))
        beta_hat = beta_hat.reshape((prior_means.shape[0], n_iter + 1))
        
        # Perform n_iter iterations of the coefficient fitting process.
        for i in tqdm(range(1, n_iter + 1)):
            
            # Shuffle the beta indexes so the order of the coefficients taking 
            # the Metropolis step is random.
            np.random.shuffle(beta_indexes)
            
            # Perform the sampling for each coefficient sequentially.
            for j in beta_indexes:
                
                # Generate a proposal beta using a normal jumping distribution 
                # and the last iteration's value.
                proposal_beta_j = (beta_hat[j, i-1] 
                                   + norm.rvs(loc=0, 
                                              scale=jumper_stds[j], 
                                              size=1))
                
                # Get a single vector for all the most recent coefficients.
                beta_now = beta_hat[:, i-1].reshape((-1, 1))
                
                # Copy the current beta vector and insert the proposal beta_j 
                # at the jth index.
                beta_prop = np.copy(beta_now)
                beta_prop[j, 0] = proposal_beta_j
                
                # Calculate the log-posterior probability of the proposed beta.
                log_p_proposal = self._log_posterior(y, X_mod, beta_prop, 
                                                    beta_now, prior_stds, 
                                                    y_std)
                # Calculate the log-posterior probability of the current beta.
                log_p_previous = self._log_posterior(y, X_mod, beta_now, 
                                                    beta_now, prior_stds, 
                                                    y_std)
                
                # Calculate the log of the r-ratio.
                log_r = log_p_proposal - log_p_previous
                
                # If r is greater than a random number from a Uniform(0, 1) 
                # distribution, insert the proposed beta_j in the list of 
                # beta_js.
                if np.log(np.random.random()) < log_r:
                    beta_hat[j, i] = proposal_beta_j
                # Otherwise, insert the old value at the jth index.
                else:
                    beta_hat[j, i] = beta_hat[j, i-1]
        
        # Set the attribute _beta_distribution with the simulated posteriors.
        self._beta_distribution = beta_hat
        # Discard the first burn_in % samples from _beta_distribution.
        self.burn(burn_in)
        # Choose the model coefficients with the median, mean, or mode of the
        # simulated posteriors of the coefficients.
        self.fit_method(method)
        # Set the _intercept_added attribute so the model remembers if it
        # needs to add an intercept to inputs for prediction.
        self._intercept_added = add_intercept
        # Set the _credible_interval attribute.
        self.credible_interval(alpha)
        
        return None