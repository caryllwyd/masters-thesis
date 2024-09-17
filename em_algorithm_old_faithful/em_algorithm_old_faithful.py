import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
os.chdir('C:/Users/turla/Documents/GitHub/masters-thesis/em_algorithm_old_faithful')
from fns_old_faithful import plot_hist_of, e_step, m_step, log_lik_mix_norm, visualise

# Loading the dataset
file_path = 'em_old_faithful.csv'  
data = pd.read_csv(file_path)
data_waiting = data["waiting"]
data_eruptions = data["eruptions"]

# Number of components
n_components = 2

# Initial parameter values
mu_0 = [60, 57]#np.random.choice(data_waiting, n_components)
sd = np.sqrt(np.var(data_waiting)) # standard deviation of dataset
sigma_0 = np.full(shape=n_components, fill_value=sd)
p_0 = np.ones(n_components) / n_components  # n coefficients of the same value

print(f"Initial means: {mu_0}")
print(f"Initial variances: {sigma_0}")
print(f"Initial mixing coefficients: {p_0}")

res = 150 # "resolution" (graininess) of density functions
psi1 = [mu_0, sigma_0, p_0]
visualise(data_waiting, mu_0, sigma_0, p_0, 
          n_iter=0, res=res, name="pics_latest_run/image0.png")
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
    visualise(data_waiting, psi2[0], psi2[1], psi2[2], n_iter=no_iter, res=res, 
                name = f"pics_latest_run/image{no_iter}.png")
    psi1 = [psi2[0], psi2[1], psi2[2]]

print(f"Number of iterations: {no_iter}")

plot_hist_of(data_waiting)
plt.savefig("histogram_old_faithful.png")
plt.close()



    