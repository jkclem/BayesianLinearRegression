# BayesianLinearRegression
This project was for me to gain a better understanding of the Metropolis-Hastings algorithm and work on my object-oriented programming skills. If you need to do any Bayesian modeling in Python, I recommend using PyMC3 (https://docs.pymc.io/). 

#### -- Project Status: [Completed]

## Project Intro/Objective
The purpose of this project was for me to work with the Metropolis-Hastings algorithm and get comfortable with object-oriented programming.

### Methods Used
* Generalized Linear Models
* Bayesian Statistics
* Metropolis-Hastings Markov Chain Monte Carlo
* Object-Oriented Programming

### Technologies
* Python (NumPy, SciPy, tqdm)

## Project Description
I created a class called MetropolisHastingsLinearModel, which is the parent class of the GaussianModel, LaplacianModel, and LogisticModel classes. It has  methods to calculate the log-prior (assuming normal priors for the distributions of the coefficients in the model), the log-posterior, fit the model/simulate the posteriors of the coefficients, burn the first x% the simulated distribution, set credible intervals, and predict for new data. For the GaussianModel and LaplacianModel classes, I only needed to add a log-likelihood method corresponding to those distributions. For the LogisticModel class, I added a method implementing the inverse-logit transformation for use in calculating the log-likelihood of the coefficients given the data and for predicting the probabilities of new observations. In addition to the method for predicting probabilities, I implemented a method for predicting the classes of new observations.