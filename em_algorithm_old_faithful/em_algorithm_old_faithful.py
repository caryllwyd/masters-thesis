import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Working directory - setup
import os
os.chdir('C:/Users/turla/Documents/GitHub/masters-thesis/em_algorithm_old_faithful')

# Load the data
file_path = 'old_faithful.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Display the first few rows of the data
print(data.head())
data_waiting = data["waiting"]

# Number of components
n_components = 2

# Initialize parameters
mu = np.random.choice(data_waiting, n_components)
sigma = np.random.random(n_components)
p = np.ones(n_components) / n_components  # Equal mixing coefficients

print(f"Initial means: {mu}")
print(f"Initial variances: {sigma}")
print(f"Initial mixing coefficients: {p}")

# E-step


# M-step

# Calculating log-likelihood


# The whole algorithm

