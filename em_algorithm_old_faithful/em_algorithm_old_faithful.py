import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize

import os
os.chdir('C:/Users/turla/Documents/GitHub/masters-thesis/em_algorithm_old_faithful')

# Loading the dataset
file_path = 'em_old_faithful.csv'  
data = pd.read_csv(file_path)
data_waiting = data["waiting"]
data_eruptions = data["eruptions"]

# function for visualising data
def visualise(mu, sigma, p, n_iter, res, name):
    """
    The function takes the values of the means mu, variances sigma and 
    weights p, and plots a histogram of data_waiting with the pdf of
    a mixture of len(p) Gaussian densities superimposed.  The image is
    then saved as a png into a dedicated folder.
    
    res controls the resolution of the pdf - how many points are used
    to render the mixture pdf.

    name represents the name of the saved image file.
    """
    n_comp = len(p)
    mu_r, sigma_r = np.round(mu, 3), np.round(sigma, 3),
    p_r = np.round(p, 3)
    text_it_legend = (
        f"Iterace č. {n_iter}"  + "\n"
        rf"$\mu$ = {mu_r}" + "\n"
        rf"$\sigma$ = {sigma_r}" + "\n"
        rf"$p$ = {p_r}")
    plt.xlabel("doba čekání (min)", labelpad=10)
    plt.tight_layout()
    plt.hist(data_waiting, color='skyblue', edgecolor='black', alpha = 0.5)
    plt.text(88, 57, text_it_legend, fontsize=8)
    # density graph
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    x_vals = np.linspace(xmin, xmax, res)
    y = np.zeros(res)

    for i in range(n_comp): # calculating mixture density
        y += p[i]*norm.pdf(x_vals, mu[i], np.sqrt(sigma[i]))
        
    legend_elements = [
        plt.Line2D([0], [0], color='r', lw=2, label='hustota směsi'),
        plt.Rectangle((0, 0), 1, 1, fc='skyblue', edgecolor='black', label='histogram dat')]
    plt.legend(handles=legend_elements, fontsize=12, loc = "upper left")

    plt.plot(x_vals, ymax / max(y) * y, linewidth=1, c="red")
    
    plt.savefig(name)
    plt.close()


# log-likelihood function (for the stopping criterion)

def log_lik_mix_norm(data, mu, sigma, p):
    """
    The function takes the dataset and all the relevant parameters (mu, 
    sigma, p) and outputs the log-likelihood of a Gaussian mixture model
    with len(p) component densities given the data.
    """
    n_comp = len(p)
    values = np.zeros((len(data), n_comp))

    for i in range(n_comp):
        values[:, i] = p[i] * norm.pdf(data, mu[i], sigma[i])  # mixture of Gaussian distributions

    return np.log(values.sum()) # constant added for numerical reasons


# the E-step

def e_step(data, mu, sigma, p):
    n_comp, n_data = len(p), len(data)
    zero_vec = np.zeros([n_comp, n_data])
    pdf, w, expectation = zero_vec, zero_vec, zero_vec

    w_sum = np.zeros(n_data)
    for i in range(n_comp):
        pdf[i, :] = norm.pdf(data, mu[i], np.sqrt(sigma[i]))
        w[i, :] = p[i] * pdf[i, :]
    w_sum = w.sum(axis=0)

    for i in range(n_comp):
        expectation[i, :] = w[i, :]/w_sum[:]
    return expectation
    
# the M-step  

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

# Initial parameter values
mu_0 = np.random.choice(data_waiting, n_components)
sd = np.sqrt(np.var(data_waiting)) # standard deviation of dataset
sigma_0 = np.full(shape=n_components, fill_value=sd)
p_0 = np.ones(n_components) / n_components  # n coefficients of the same value

print(f"Initial means: {mu_0}")
print(f"Initial variances: {sigma_0}")
print(f"Initial mixing coefficients: {p_0}")


psi1 = [mu_0, sigma_0, p_0]
visualise(mu_0, sigma_0, p_0,
          n_iter=0, res=100, name="pics_latest_run/image0.png")
like_diff, no_iter = 1, 0
while(like_diff > 1e-6):
    no_iter += 1 # iteration count
    psi2 = m_step(e_step(data_waiting, psi1[0], psi1[1], psi1[2]), 
                  data_waiting, psi1[2])
    l1 = log_lik_mix_norm(data_waiting, psi1[0], psi1[1], psi1[2])
    l2 = log_lik_mix_norm(data_waiting, psi2[0], psi2[1], psi2[2])
    like_diff = abs(l2 - l1)
    if (no_iter % 5 == 0):
        print(f"mu({no_iter}): {psi2[0]},", 
              f"like: {log_lik_mix_norm(data_waiting, psi2[0], psi2[1], psi2[2])}")
        visualise(psi2[0], psi2[1], psi2[2], n_iter=no_iter, res = 100, 
                  name = f"pics_latest_run/image{no_iter}.png")
    psi1 = [psi2[0], psi2[1], psi2[2]]
    #visualise(psi2[0], psi2[1], psi2[2], 100, f"pics_latest_run/image{no_iter}.png")

    
#s = em_algorithm(data_waiting, mu, sigma, p)
print(f"Number of iterations: {no_iter}")
#print(f"{s}")
