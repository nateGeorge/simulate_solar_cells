"""
Here are some parameters from the manufacturing of CIGS solar cells.

- Constituent metals atomic percents (Cu, In, Ga) from XRF (https://en.wikipedia.org/wiki/X-ray_fluorescence)
- Selenium flowrate in reactor
- Overall thickness
- CdSe thickness
"""

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

SAMPLE_SIZE = 5000
# using stoichiometry of Cu(In,Ga)3Se5 assuming approximately 1:2 In:Ga

cu_ideal_pct = 1/9
in_ideal_pct = 1/9
ga_ideal_pct = 2/9
se_ideal_pct = 5/9

ideal_se_flowrate = 12
ideal_thickness = 600

avg_efficiency_factor = 13

def generate_metal_dist(ideal_pct):
    """
    generates distribution of metals

    args:
    ideal_pct: float; ideal atomic percentage of element
    """
    offset = np.random.random() / 20 - 0.025 # between -0.025 and 0.025
    dist = st.t.rvs(df=100, loc=ideal_pct + offset, scale=0.01, size=SAMPLE_SIZE, random_state=42)
    # add random noise
    dist += st.laplace.rvs(loc=0, scale=0.01, size=SAMPLE_SIZE, random_state=42)
    return dist

np.random.seed(42)
cu_dist = generate_metal_dist(cu_ideal_pct)

# check how distribution looks
# plt.hist(cu_dist, bins=30)
# plt.show()

in_dist = generate_metal_dist(in_ideal_pct)
ga_dist = generate_metal_dist(ga_ideal_pct)

# should be about 4/9 = 0.44 on avg -- pretty close
# metal_sums = cu_dist + in_dist + ga_dist
# print(np.mean(metal_sums))

# generate selenium flowrates
offset = np.random.random() / 5 - 0.1 # between -0.1 and 0.1
se_flowrates = st.norm.rvs(loc=ideal_se_flowrate + offset, scale=0.3, size=SAMPLE_SIZE, random_state=42)
# add noise
se_flowrates += st.norm.rvs(loc=0, scale=0.5, size=SAMPLE_SIZE, random_state=42)

# plt.hist(se_flowrates, bins=30)
# plt.show()


# generate overall thickness, in nanometers
offset = np.random.random() * 20 - 10 # between -10 and 10
thickness = st.norm.rvs(loc=ideal_thickness, scale=10, size=SAMPLE_SIZE, random_state=42)

# add noise
thickness += st.lognorm.rvs(s=1, scale=10, size=SAMPLE_SIZE, random_state=42)

# look at distribution
# plt.hist(thickness, bins=30)
# plt.show()

# generate efficiencies; idea is there are optimal values for variables, and they interact non-linearly
cu_factor = (1 + (cu_dist - cu_ideal_pct) ** 2)
cu_factor /= np.mean(cu_factor)
in_factor = (1 + (in_dist - in_ideal_pct) ** 2)
in_factor /= np.mean(in_factor)
ga_factor = (1 + (ga_dist - ga_ideal_pct)) ** 2
ga_factor /= np.mean(ga_factor)

se_factor = np.exp((1 + (se_flowrates - ideal_se_flowrate) / 40))
se_factor /= np.mean(se_factor)
thickness_factor =  np.tanh(1 + (thickness - ideal_thickness) / 10000)
thickness_factor /= np.mean(thickness_factor)
eff = avg_efficiency_factor * (3 + cu_factor * in_factor * ga_factor * se_factor * thickness_factor) - (cu_factor + in_factor * ga_factor) - (se_factor * thickness_factor) ** 2)

# eff = avg_efficiency_factor * (np.log(1 + cu_factor ** 3 * in_factor ** 2 * thickness_factor ** 3) \
#                             + 0.5 * np.log(1 + cu_factor ** 2 * se_factor + in_factor * se_factor ** 3) \
#                             + 0.3 * np.exp(cu_factor * thickness_factor ** 0.5)) / (np.log(in_factor * cu_factor + ga_factor * se_factor ** 3) + se_factor * thickness_factor)

# remove outliiers and replace with random numbers
# outliers = np.where(eff > 15)[0]
# eff[outliers] = st.laplace.rvs(loc=12, scale=0.5, size=outliers.shape[0])
# add noise
eff += st.norm.rvs(loc=0, scale=0.3, size=SAMPLE_SIZE, random_state=42)
plt.hist(eff, bins=30)
plt.show()

plt.scatter(cu_dist, eff)
plt.show()

plt.scatter(in_dist, eff)
plt.show()

plt.scatter(ga_dist, eff)
plt.show()

plt.scatter(se_flowrates, eff)
plt.show()

plt.scatter(thickness, eff)
plt.show()
