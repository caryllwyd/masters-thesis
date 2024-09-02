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



# log-likelihood function

def log_lik_two_norm(data, mu, sigma, p):
    values = np.zeros((len(data), n_components))

    for i in range(n_components):
        values[:, i] = p[i] * norm.pdf(data, mu[i], sigma[i])  # mixture of Gaussian distributions

    return values.sum()

# EM iterative algorithm function

def em_algorithm(data, mu_0, sigma_0, p_0, n_components):
    tau = np.zeros((len(data), n_components)) # zvazit, zda tam dat pocatecni odhad
    mu_k = mu_0
    sigma_k = sigma_0
    p_k = p_0

    #while(log_lik_two_norm(data_waiting, mu_k, sigma_k, p_k) < 1e-2):
        #tau = tau

    return mu_k, sigma_k, p_k, tau


# E-step
#def e_step_tau(data, mu_k, sigma_k, p_k):
    

# M-step
#def m_step(data, values):
   



# Number of components
n_components = 2

# Initialise parameters
mu = np.random.choice(data_waiting, n_components)
sigma = 1 + 2*np.random.random(n_components)
p = np.ones(n_components) / n_components  # n coefficients of the same value

print(f"Initial means: {mu}")
print(f"Initial variances: {sigma}")
print(f"Initial mixing coefficients: {p}")

like = log_lik_two_norm(data_waiting, mu, sigma, p)

# Visualisation
plt.scatter(data_waiting, data_eruptions, c="cyan", s=8, edgecolors="black")

# PDF graph
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
x_vals = np.linspace(xmin, xmax, 100)
y_1 = norm.pdf(x_vals, mu[0], sigma[0])
y_2 = norm.pdf(x_vals, mu[1], sigma[1])

plt.plot(x_vals, ymax*y_1, linewidth=2, c="red")
plt.plot(x_vals, ymax*y_2, linewidth=2, c="blue")
plt.show()

s = em_algorithm(data_waiting, mu, sigma, p, n_components=2)

print(f"{s}")
