# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# Simulate some fake data:

import matplotlib.pyplot as plt

# +
import numpy as np

random = np.random.default_rng(5430)

true_sigma = 1.2
num_targ = 1000
semiamp = np.exp(random.uniform(np.log(0.01), np.log(10.0), num_targ))
period = np.exp(random.uniform(np.log(1.0), np.log(100.0), num_targ))
phi = random.uniform(0, 2 * np.pi, num_targ)
num_transit = random.integers(3, 25, num_targ)
sample_variance = np.empty_like(phi)
expected = np.empty_like(phi)

for k in range(num_targ):
    t = random.uniform(0, 600, num_transit[k])
    y = semiamp[k] * np.sin(2 * np.pi * t / period[k] + phi[k])
    noise = np.exp(np.log(true_sigma) + 0.1 * random.normal()) * random.normal(
        size=num_transit[k]
    )
    y += noise
    sample_variance[k] = np.var(y, ddof=1)
    expected[k] = np.var(noise, ddof=1)

plt.figure(figsize=(6, 4))
_, bins, _ = plt.hist(
    np.log10(sample_variance), 20, color="k", histtype="step", label="observed"
)
plt.hist(
    np.log10(expected),
    bins,
    color="k",
    histtype="stepfilled",
    alpha=0.2,
    linewidth=0,
    label="expected",
)
plt.axvline(
    np.log10(true_sigma ** 2), color="C1", lw=1, label="true $\sigma^2$"
)
plt.legend(loc=2)
plt.yscale("log")
plt.ylabel("number of simulated targets")
plt.xlabel("observed $\log_{10}(s^2)$")
plt.savefig("infer_sigma_data.pdf", bbox_inches="tight")
# +
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.config import config as jax_config
from numpyro.distributions.transforms import AffineTransform
from numpyro_ncx2 import NoncentralChi2

jax_config.update("jax_enable_x64", True)


def model(num_transit, statistic=None):
    log_sigma = numpyro.sample("log_sigma", dist.Normal(0.0, 10.0))

    with numpyro.plate("targets", len(num_transit)):
        log_k = numpyro.sample("log_k", dist.Normal(0.0, 10.0))
        lam = num_transit * 0.5 * jnp.exp(2 * (log_k - log_sigma))
        numpyro.sample(
            "obs",
            dist.TransformedDistribution(
                NoncentralChi2(num_transit, lam),
                AffineTransform(loc=0.0, scale=jnp.exp(2 * log_sigma)),
            ),
            obs=statistic,
        )


init = {
    "log_sigma": 0.5 * np.log(np.median(sample_variance)),
    "log_k": np.log(np.sqrt(sample_variance)),
}
guide = numpyro.infer.autoguide.AutoNormal(
    model, init_loc_fn=numpyro.infer.init_to_value(values=init)
)
optimizer = numpyro.optim.Adam(step_size=1e-3)
svi = numpyro.infer.SVI(
    model, guide, optimizer, loss=numpyro.infer.Trace_ELBO()
)
svi_result = svi.run(
    jax.random.PRNGKey(8596),
    20_000,
    num_transit,
    statistic=(num_transit - 1) * sample_variance,
    progress_bar=False,
)
# -

plt.figure(figsize=(6, 4))
factor = np.log10(np.exp(1))
plt.errorbar(
    np.log10(semiamp),
    factor * svi_result.params["log_k_auto_loc"],
    yerr=factor * svi_result.params["log_k_auto_scale"],
    fmt=",k",
    alpha=0.2,
)
plt.plot(
    np.log10(semiamp),
    factor * svi_result.params["log_k_auto_loc"],
    ".k",
    ms=3,
    alpha=0.5,
)
x = np.log10([0.2, 1.2 * semiamp.max()])
plt.plot(x, x, "k", lw=0.5)
plt.axhline(
    factor * svi_result.params["log_sigma_auto_loc"],
    color="C0",
    lw=2,
    label="inferred $\sigma$",
)
plt.axhline(
    np.log10(true_sigma), ls="dashed", color="C1", lw=2, label="true $\sigma$"
)
plt.xlim(x.min(), x.max())
plt.ylim(-0.5, plt.ylim()[1])
plt.legend()
plt.xlabel("$\log_{10}$(simulated rv semi-amplitude) [km/s]")
plt.ylabel("$\log_{10}$(inferred rv semi-amplitude) [km/s]")
plt.savefig("infer_sigma.pdf", bbox_inches="tight")
