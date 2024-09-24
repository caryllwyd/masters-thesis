import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm

# ZIP distribution graph

# parameters of the mixture
par_dg = 0  # parameter of the degenerate distribution
w_dg = 4/24  # weight of the degenerate distribution

par_poi = 3  # lambda of the Poisson distribution
w_poi = 20/24  # weight of the Poisson distribution

x_values = np.arange(0, 12) # integers from 0 to 11


# probability mass functions 
pmf_dg = np.zeros_like(x_values, dtype=float)  # pmf of dg - zeroes
pmf_dg[par_dg] = 1  # pmf of dg - where the "1" is

pmf_poi = poisson.pmf(x_values, par_poi)  # pmf of Poi

pmf_mix = w_poi * pmf_poi + w_dg * pmf_dg # pmf of mixture

# Plot the PMFs
plt.figure(dpi=300)
(markerline, stemlines, baseline) = plt.stem(x_values+0.1, pmf_mix,
                                             "deepskyblue", 
                                             markerfmt="deepskyblue", 
                                             basefmt="")
plt.setp(baseline, visible=False)

(markerline, stemlines, baseline) = plt.stem(x_values-0.1, pmf_poi, 
                                             "salmon", 
                                             markerfmt="salmon", 
                                             basefmt="")
plt.setp(baseline, visible=False)

plt.xticks(x_values)
plt.xlabel("počet hovorů za hodinu")
plt.ylabel("pravděpodobnostní funkce")

legend_elements = [
        plt.Line2D([0], [0], color="deepskyblue", lw=2, label="směs rozdělení"),
        plt.Line2D([0], [0], color="salmon", lw=2, label="Poissonovo rozdělení")]
plt.legend(handles=legend_elements, fontsize=10, loc = "upper right")

plt.savefig("zip_distribution.png")


# Mixture of normal distributions.
def plot_mixture_of_normals(means, variances, weights, colors, filename,
                            leg_loc = "upper left",
                            res=200, x_range=(-10, 10)):
    """

    """
    
    # Convert variances to standard deviations (since norm.pdf uses standard deviation)
    std_devs = np.sqrt(variances)
    
    # Define the range of x values (continuous)
    x_values = np.linspace(x_range[0], x_range[1], res)
    
    # Initialize the mixture PDF to zero
    pdf_mix = np.zeros_like(x_values)
    
    # Plot individual normal distributions and accumulate their weighted PDFs
    plt.figure(dpi=300)
    for i in range(len(means)):
        # PDF of each normal distribution
        pdf_i = norm.pdf(x_values, means[i], std_devs[i])
        # Weighted PDF of the current distribution
        pdf_mix += weights[i] * pdf_i
        # Plot the current normal distribution
        plt.plot(x_values, pdf_i,  label=f"rozdělení č. {i+1}",
                 color = colors[i])
    
    # Plot the mixture of normals
    plt.plot(x_values, pdf_mix, label="směs hustot", color="deepskyblue")
    
    # Adding labels and title
    plt.xlabel("x")
    plt.ylabel("hustota rozdělení")
    
    # Add legend
    plt.legend(loc=leg_loc, fontsize=10)
    
    # Save the plot
    plt.savefig(filename)
    

means = [0, 3,]  # List of means
variances = [1, 1.5]  # List of variances (std^2)
weights = [0.4, 0.6]  # Weights (should sum to 1)
colors = ["salmon", "red"]

plot_mixture_of_normals(means, variances, weights, colors = colors,
                         x_range=(-5, 8), filename="test_mixture.png")