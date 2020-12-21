# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 16:04:49 2020

@author: jkcle
"""
import numpy as np
from scipy.stats import norm, mode
from tqdm import tqdm

class MHLinearRegressor():
    """ 
    This is a parent class for linear models whose coefficients are fit using
    a Metropolis-Hastings Markov Chain Monte Carlo algorithm with a normal
    prior for the distribution of the coefficients.
    """
    
    def __init__(self):
        self._intercept_added = False
        self._beta_distribution = np.empty(1)
        self._beta_hat = np.empty(1)
        self._credible_intervals = np.empty(1)
        
        
    def _normal_log_prior(self, beta, prior_means, prior_stds):
        """
        This method calculates the log-prior of the coefficients using our
        normal priors. The log-likelihood plus the log-prior is proportional 
        to the log-posterior. It is used in the model fitting process.

        Parameters
        ----------
        beta : numpy array
            A 1-D vector of coefficients in the regression model.
        prior_means : numpy array
            A 1-D vector of means for the prior distributions of the 
            coefficients. 
        prior_stds : numpy array
            A 1-D vector of standard devations for the prior distributions of
            the coefficients. 

        Returns
        -------
        log_prior : float
            A value for the log-prior.

        """
        
        # Find log-densities for each coefficent given their priors.
        log_prior_densities = norm.logpdf(beta, 
                                          loc=prior_means.reshape((-1, 1)),
                                          scale=prior_stds.reshape((-1, 1)))
        # Sum the log-densities.
        log_prior = np.sum(log_prior_densities)
        
        return log_prior
    
    
    def _log_likelihood(self, y, X, beta):
        """
        This method needs to be overwritten by the child classes because it
        depends on the distribution of the data being modeled. It calculates 
        the log-likelihood of the betas given the data. The log-likelihood 
        plus the log-prior is proportional to the log-posterior. It is used in 
        the model fitting process.

        Parameters
        ----------
        y : numpy array
            A 1-D array of the endogenous variable (target variable).
        X : numpy array
            A 2-D matrix where rows represent observations and columns 
            represent variables.
        beta : numpy array
            A 1-D vector of coefficients in the regression model.

        Returns
        -------
        None.

        """      
        pass
    
    
    def _log_posterior(self, y, X, beta, prior_means, prior_stds):
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

        Returns
        -------
        _log_posterior : float
            A value proportional to the log-posterior.

        """   
        
        # Calculate a value proportional to the log-posterior.
        _log_posterior = (self._normal_log_prior(beta, prior_means, prior_stds) 
                         + self._log_likelihood(y, X, beta))
        
        return _log_posterior
    
    
    def fit(self, 
            y,
            X, 
            prior_means,
            prior_stds,
            jumper_stds, 
            n_iter=10000,
            add_intercept=True,
            method="median",
            burn_in=0.1,
            alpha=0.05,
            random_seed=1):
        """
        This method fits a model to the input data by simulating the posterior
        distributions of the coefficients using the Metropolis-Hastings 
        algorithm.

        Parameters
        ----------
        y : numpy array
            A 1-D array of the endogenous variable (target variable).
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
                                                    beta_now, prior_stds)
                # Calculate the log-posterior probability of the current beta.
                log_p_previous = self._log_posterior(y, X_mod, beta_now, 
                                                    beta_now, prior_stds)
                
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
        self._burn(burn_in)
        # Choose the model coefficients with the median, mean, or mode of the
        # simulated posteriors of the coefficients.
        self.fit_method(method)
        # Set the _intercept_added attribute so the model remembers if it
        # needs to add an intercept to inputs for prediction.
        self._intercept_added = add_intercept
        # Set the _credible_interval attribute.
        self._credible_interval(alpha)
        
        return None
      
    
    def fit_method(self, method):
        """
        This method sets the _beta_hat attribute as either the median, mean,
        or mode as the estimate of each coefficient in the beta vector.

        Parameters
        ----------
        method : str
            Determines whether the _beta_hat attribute uses either the median,
            mean, or mode of the _beta_distribution attribute as the estimates
            of the regression coefficients.

        Returns
        -------
        None.

        """
        
        assert ((method == "median")
                | (method == "mean")
                | (method == "mode")), "method must be median, mean, or mode."
        
        # Set the beta_hat array as either the median, mean, or mode of the
        # _beta_distribution attribute.
        if method == "median":
            beta_hat = np.median(self._beta_distribution, 
                                 axis=1).reshape((-1,1))
        elif method == "mean":
            beta_hat = np.mean(self._beta_distribution, 
                               axis=1).reshape((-1,1))
        else:
            beta_hat = mode(self._beta_distribution, 
                            axis=1)[0]
        
        # Sets the _beta_hat attribute using the chosen method.
        self._beta_hat = beta_hat
        
        return None
        
    
    def _burn(self, burn_in):
        """
        This function discards the first 100*burn_in % of the simulated
        posterior distribution of beta.

        Parameters
        ----------
        burn_in : TYPE
            Discards the first burn_in % of samples.

        Returns
        -------
        None.

        """
        
        assert (burn_in > 0) & (burn_in < 1), "burn_in must be in (0, 1)."
        
        # Set the index for the start of the distributions of the coefficients
        # burn_in % of the way into the raw distributions adding 1 for the 
        # prior.
        start = round(burn_in * self._beta_distribution.shape[1]) + 1

        # Set the _beta_distribution attribute without the burn-in samples.
        self._beta_distribution = self._beta_distribution[:, start:]
        
        return None
     
    
    def _credible_interval(self, alpha):
        """
        This method sets the _credible_interval attribute as the 
        100*(1-alpha)% credible interval for each coefficient in the beta 
        vector.
        

        Parameters
        ----------
        alpha : float, optional
            Determines the credible intervals widths.

        Returns
        -------
        None.

        """
        
        assert (alpha > 0) & (alpha < 1), "alpha must be in (0, 1)."
        
        # Calculate the 100*(1-alpha)% credible intervals for each coefficient.
        credible_intervals = np.transpose(np.quantile(self._beta_distribution,
                                                      q=(alpha/2, 1-alpha/2),
                                                      axis=1))
        # Set the _credible_intervals attribute to the calculated credible 
        # intervals.
        self._credible_intervals = credible_intervals
        
        return None
        
    
    def predict(self, X):
        """
        This method returns the predictions of a new data matrix matching the
        training matrix in column number and ordering from the fit model.

        Parameters
        ----------
        X : numpy array
            A 2-D matrix where rows represent observations and columns 
            represent variables.

        Returns
        -------
        predictions : numpy array
            A 1-D vector of the model's predictions.

        """
        
        # Add a column of ones in the first column if this instance of the
        # class is fit with an intercept.
        if self._intercept_added:
            X_new = np.append(np.ones(shape=(X.shape[0], 1)), X, 1)
        # Do not add a column of ones if it was not added by this instance.
        else:
            X_new = X
        
        # Calculate the predictions by multiplying the new observation matrix
        # by the vector of model coefficients.
        predictions = np.matmul(X_new, self._beta_hat)
            
        return predictions