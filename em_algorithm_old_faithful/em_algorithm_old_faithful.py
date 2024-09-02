import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Setting up working directory
import os
os.chdir('C:/Users/turla/Documents/GitHub/masters-thesis/em_algorithm_old_faithful')

# Loading data
file_path = 'old_faithful.csv'  # Replace with your file path
data = pd.read_csv(file_path)

data_waiting = data["waiting"]
data_eruptions = data["eruptions"]
zeroes = np.zeros(len(data_waiting))

# Visualisation
plt.scatter(data_waiting, data_eruptions, c="cyan", s=6, edgecolors="black")
plt.show()


# Number of components
n_components = 2

# Initialise parameters
mu = np.random.choice(data_waiting, n_components)
sigma = np.random.random(n_components)
p = np.ones(n_components) / n_components  # n coefficients of the same value

print(f"Initial means: {mu}")
print(f"Initial variances: {sigma}")
print(f"Initial mixing coefficients: {p}")

# E-step
def e_step(data, mu_k, sigma_k, p_k):
    values = np.zeros((len(data), n_components))

    for i in range(n_components):
        values[:, i] = p_k[i] * norm.pdf(data, mu[i], sigma[i])  # mixture of Gaussian distributions

    values = values / values.sum(axis=1, keepdims=True)
    return values

# M-step
def m_step(data, values):
    n_kp = values.sum(axis=0) # n_k+1
    mu = (values.T @ data) / n_kp
    sigma = np.sqrt((values.T @ (data - mu)**2) / n_kp)
    p_kp = n_kp / len(data)
    return mu, sigma, p_kp

# Calculating log-likelihood


# The whole algorithm

