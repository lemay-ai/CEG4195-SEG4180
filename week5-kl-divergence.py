# container: pytorch
# region: kl
import torch
from torch.distributions import Normal, kl_divergence

# Define two distributions
dist1 = Normal(loc=0.0, scale=1.0)  # Normal distribution with mean 0 and std 1
dist2 = Normal(loc=1.0, scale=2.0)  # Normal distribution with mean 1 and std 2

# Calculate the KL divergence (exact, for two normals)
kl = kl_divergence(dist1, dist2)

# Print the KL divergence
print(f"KL Divergence: {kl:.3f}")

# Sample from the distributions and compute KL divergence using Monte Carlo sampling
samples1 = dist1.sample((10000,))
samples2 = dist2.sample((10000,))
log_prob1 = dist1.log_prob(samples1)
log_prob2 = dist2.log_prob(samples1)

monte_carlo_kl = (log_prob1 - log_prob2).mean()
print(f"Monte Carlo KL Divergence: {monte_carlo_kl:.3f}")
# endregion

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(samples1, bins=50, alpha=0.5, label='samples_dist1', density=True)
plt.hist(samples2, bins=50, alpha=0.5, label='samples_dist2', density=True)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Distribution of dist1 and dist2 samples')
plt.legend()
plt.show()
