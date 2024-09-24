import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
os.chdir('C:/Users/turla/Documents/GitHub/masters-thesis/em_algorithm_old_faithful')

file_path = 'em_old_faithful.csv'  
data = pd.read_csv(file_path)
data_waiting = data["waiting"]
data_eruptions = data["eruptions"]

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

def multiple_runs_em(data, no_runs):
    n_components = 2
    mu_hat = np.zeros([no_runs, n_components])
    sigma_hat = np.zeros([no_runs, n_components])
    p_hat = np.zeros([no_runs, n_components])
    like = np.zeros(no_runs)
    for i in range(no_runs):
        mu_0 = np.random.choice(data, n_components)
        sd = np.sqrt(np.var(data)) # standard deviation of dataset
        sigma_0 = np.full(shape=n_components, fill_value=sd)
        p_0 = np.ones(n_components) / n_components  # n coefficients of the same value
        psi1 = [mu_0, sigma_0, p_0]

        like_diff, no_iter = 1, 0
        while(like_diff > 1e-6):
            no_iter += 1 # iteration count
            psi2 = m_step(e_step(data, psi1[0], psi1[1], psi1[2]), 
                        data, psi1[2])
            l1 = log_lik_mix_norm(data, psi1[0], psi1[1], psi1[2])
            l2 = log_lik_mix_norm(data, psi2[0], psi2[1], psi2[2])
            like_diff = abs(l2 - l1)
            psi1 = [psi2[0], psi2[1], psi2[2]]

        mu_hat[i, :], sigma_hat[i, :], p_hat[i, :] = np.round(psi1, 3)
        like[i] = np.round(l2, 5)
    for j in range(no_runs):
        print(f"mu_hat: {mu_hat[j, :]}, likelihood: {like[j]}")



def plot_hist_of(data_waiting):
    plt.figure(dpi=300)
    plt.xlabel("doba čekání (min)", labelpad=10)
    plt.ylabel("Frekvence výskytu", labelpad=10)
    plt.tight_layout()
    bin_ticks = range(42, 102, 5)
    plt.hist(data_waiting, color='skyblue', edgecolor='black', alpha=0.5, bins=bin_ticks) 
    plt.xticks(bin_ticks)
    
def visualise(data, mu, sigma, p, n_iter, res, name):
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
    plot_hist_of(data)
    
    # density graph
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    x_vals = np.linspace(xmin, xmax, res)
    y = np.zeros(res)

    for i in range(n_comp): # calculating mixture density
        y += p[i]*norm.pdf(x_vals, mu[i], np.sqrt(sigma[i]))

    density = ymax / max(y) * y

    text_text = (
        f"Iterace č. {n_iter}"  + "\n"
        rf"$\mu$ = {mu_r}" + "\n"
        rf"$\sigma$ = {sigma_r}" + "\n"
        rf"$p$ = {p_r}")
    
    plt.text(38, 45, text_text, fontsize=8)
    
        
    legend_elements = [
        plt.Line2D([0], [0], color='r', lw=2, label='hustota směsi'),
        plt.Rectangle((0, 0), 1, 1, fc='skyblue', edgecolor='black', label='histogram dat')]
    plt.legend(handles=legend_elements, fontsize=10, loc = "upper left")
    
    plt.plot(x_vals, density, linewidth=1, c="red")
   
    plt.savefig(name)
    plt.close()   

multiple_runs_em(data_waiting, 100)