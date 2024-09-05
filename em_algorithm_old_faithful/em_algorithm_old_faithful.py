import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize

# Setting up working directory
import os
os.chdir('C:/Users/turla/Documents/GitHub/masters-thesis/em_algorithm_old_faithful')

# Loading data
file_path = 'old_faithful.csv'  # Replace with your file path
data = pd.read_csv(file_path)

data_waiting = data["waiting"]
data_eruptions = data["eruptions"]




# log-likelihood function

def log_lik_mix_norm(data, mu, sigma, p):
    n_comp = len(p)
    values = np.zeros((len(data), n_comp))

    for i in range(n_comp):
        values[:, i] = p[i] * norm.pdf(data, mu[i], sigma[i])  # mixture of Gaussian distributions

    return np.log(values.sum()+1e-10)


# EM iterative algorithm function

def em_algorithm(data, mu_0, sigma_0, p_0):
    n_comp = len(p_0) # number of mixt. comps
    n_data = len(data) # length of dataset
     
    # initial values:
    mu_k = mu_0[:] 
    sigma_k = sigma_0[:]
    p_k = p_0[:]
    likef = np.zeros(n_data)
    likef_i = np.zeros((n_data,n_comp))
    tau = np.zeros((n_data,n_comp))

    print(f"abs likelihood: {np.abs(log_lik_mix_norm(data_waiting, mu_k, sigma_k, p_k))}")

    while(np.abs(log_lik_mix_norm(data, mu_k, sigma_k, p_k)) > 0.1): 
        for i in range(n_comp):
            likef_i[:, i] = norm.pdf(data, mu_k[i], sigma_k[i]) # likelihood values
        for i in range(n_data):
            likef[i] = likef_i[i, :] @ p_k
        #print(f"f: {likef}")
        for i in range(n_comp):
            tau[:, i] = p_k[i] * (likef_i[:, i]/likef[:])
            p_k[i] = tau[:, i].sum()/n_data
        mu_k = (tau.T @ data)/n_data  
        #sigma_k = (tau.T @ ((mu_k - data)^2))/n_data 
        #print(f"tau: {tau}") 
        print(f"abs. likelihood after update: {np.abs(log_lik_mix_norm(data, mu_k, sigma_k, p_k))}")
        print(f"Mu_k: {mu_k}")

    return mu_k, sigma_k, p_k, tau[0:5, :]






# Number of components
n_components = 2

# Initialise parameters
mu = np.random.choice(data_waiting, n_components)
sigma = 1 + 2*np.random.random(n_components)
p = np.ones(n_components) / n_components  # n coefficients of the same value

print(f"Initial means: {mu}")
print(f"Initial variances: {sigma}")
print(f"Initial mixing coefficients: {p}")

like = log_lik_mix_norm(data_waiting, mu, sigma, p)

# Visualisation
zeroes = np.zeros(len(data_waiting)) # for pdf's
plt.scatter(data_waiting, data_eruptions, c="cyan", s=8, edgecolors="black")

# PDF graph
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
x_vals = np.linspace(xmin, xmax, 100)
y_1 = norm.pdf(x_vals, mu[0], sigma[0])
y_2 = norm.pdf(x_vals, mu[1], sigma[1])

plt.plot(x_vals, ymax*y_1, linewidth=1, c="red")
plt.plot(x_vals, ymax*y_2, linewidth=1, c="blue")
#plt.show()

s = em_algorithm(data_waiting, mu, sigma, p)

print(f"{s}")
