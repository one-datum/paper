import numpy as np
import matplotlib.pyplot as plt

random = np.random.default_rng(5430)

true_sigma = 1.2
num_targ = 1000
semiamp = np.exp(random.uniform(np.log(0.01), np.log(10.0), num_targ))
period = np.exp(random.uniform(np.log(1.0), np.log(100.0), num_targ))
phi = random.uniform(0, 2 * np.pi, num_targ)
num_obs = random.integers(3, 25, num_targ)
obs_s2 = np.empty_like(phi)
base_s2 = np.empty_like(phi)

for k in range(num_targ):
    t = random.uniform(0, 600, num_obs[k])
    y = semiamp[k] * np.sin(2 * np.pi * t / period[k] + phi[k])
    noise = np.exp(np.log(true_sigma) + 0.1 * random.normal()) * random.normal(
        size=num_obs[k]
    )
    y += noise
    obs_s2[k] = np.var(y, ddof=1)
    base_s2[k] = np.var(noise, ddof=1)

plt.figure(figsize=(6, 4))
_, bins, _ = plt.hist(
    np.log10(obs_s2), 20, color="k", histtype="step", label="observed"
)
plt.hist(
    np.log10(base_s2),
    bins,
    color="k",
    histtype="stepfilled",
    alpha=0.2,
    linewidth=0,
    label="expected",
)
plt.axvline(np.log10(true_sigma ** 2), color="C1", lw=1, label="true $\sigma^2$")
plt.legend(loc=2)
plt.yscale("log")
plt.ylabel("number of simulated targets")
plt.xlabel("observed $\log_{10}(s^2)$")
plt.savefig("infer_sigma.pdf", bbox_inches="tight")
