import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize

# Setting up working directory
import os
os.chdir('C:/Users/turla/Documents/GitHub/masters-thesis/em_algorithm_old_faithful')

# Loading data
file_path = 'em_old_faithful.csv'  # Replace with your file path
data = pd.read_csv(file_path)

data_waiting = data["waiting"]
data_eruptions = data["eruptions"]


# log-likelihood function

def log_lik_mix_norm(data, mu, sigma, p):
    n_comp = len(p)
    values = np.zeros((len(data), n_comp))

    for i in range(n_comp):
        values[:, i] = p[i] * norm.pdf(data, mu[i], sigma[i])  # mixture of Gaussian distributions

    return np.log(values.sum()) # constant added for numerical reasons




def e_step(data, mu, sigma, p):
    n_comp, n_data = len(p), len(data)
    pdf, w, expectation = np.zeros([n_comp, n_data]), np.zeros([n_comp, n_data]), np.zeros([n_comp, n_data])
    w_sum = np.zeros(n_data)
    for i in range(n_comp):
        pdf[i, :] = norm.pdf(data, mu[i], np.sqrt(sigma[i]))
        w[i, :] = p[i] * pdf[i, :]
    w_sum = w.sum(axis=0)

    for i in range(n_comp):
        expectation[i, :] = w[i, :]/w_sum[:]
    return expectation
    

def m_step(expectation, data, p):
    n_comp, n_data = len(p), len(data)
    p_hat = np.zeros(n_comp)
    mu_hat_num, mu_hat_den, mu_hat = np.zeros(n_comp), np.zeros(n_comp), np.zeros(n_comp)

    for i in range(n_comp):
        for j in range(n_data):
            mu_hat_num[i] += expectation[i, j] * data[j]
            mu_hat_den[i] += expectation[i, j]
        mu_hat[i] = mu_hat_num[i]/mu_hat_den[i]
    
    sigma_hat_num, sigma_hat_den, sigma_hat = np.zeros(n_comp), np.zeros(n_comp), np.zeros(n_comp)
    
    for i in range(n_comp):
        for j in range(n_data):
            sigma_hat_num[i] += expectation[i, j] * ((mu_hat[i] - data[j])**2)
            sigma_hat_den[i] += expectation[i, j]
        sigma_hat[i] = sigma_hat_num[i]/sigma_hat_den[i]

    p_hat = expectation.sum(axis=1)/n_data
    return mu_hat, sigma_hat, p_hat  

# Number of components
n_components = 2

# Initialise parameters
mu_0 = np.random.choice(data_waiting, n_components)
sigma_0 = 1 + 10*np.random.random(n_components)
p_0 = np.ones(n_components) / n_components  # n coefficients of the same value

print(f"Initial means: {mu_0}")
print(f"Initial variances: {sigma_0}")
print(f"Initial mixing coefficients: {p_0}")


# Visualisation

def visualise(mu, sigma, p, res, name):
    n_comp = len(p)
    plt.hist(data_waiting, color='skyblue', edgecolor='black', alpha = 0.5)
    #plt.scatter(data_waiting, data_eruptions, c="cyan", s=8, edgecolors="black")

# PDF graph
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    x_vals = np.linspace(xmin, xmax, res)
    y = np.zeros(res)
    for i in range(n_comp):
        y += p[i]*norm.pdf(x_vals, mu[i], np.sqrt(sigma[i]))

    plt.plot(x_vals, ymax / max(y) * y, linewidth=1, c="red")
    plt.savefig(name)
    plt.close()

#e = e_step(data_waiting, mu, sigma, p)
#m = m_step(e, data_waiting, p)

psi1 = [mu_0, sigma_0, p_0]
visualise(mu_0, sigma_0, p_0, 1000, "pics_latest_run/image0.png")
like_diff, no_iter = 1, 0
while(like_diff > 1e-5):
    no_iter += 1
    psi2 = m_step(e_step(data_waiting, psi1[0], psi1[1], psi1[2]), data_waiting, psi1[2])

    l1 = log_lik_mix_norm(data_waiting, psi1[0], psi1[1], psi1[2])
    l2 = log_lik_mix_norm(data_waiting, psi2[0], psi2[1], psi2[2])
    like_diff = abs(l2 - l1)
    if (no_iter % 5 == 0):
        print(f"mu({no_iter}): {psi2[0]}, like: {log_lik_mix_norm(data_waiting, psi2[0], psi2[1], psi2[2])}")
    psi1 = [psi2[0], psi2[1], psi2[2]]
    visualise(psi1[0], psi1[1], psi2[2], 100, f"pics_latest_run/image{no_iter}.png")

    
#s = em_algorithm(data_waiting, mu, sigma, p)
print(f"no_iter: {no_iter}")
#print(f"{s}")
