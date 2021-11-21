#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

THRESHOLD = 0.001

with fits.open("../archive/inference/inferred.fits.gz") as f:
    data = f[1].data

conf = data["rv_unc_conf"]
sigma = data["rv_est_uncert"]
nb_transits = data["dr2_rv_nb_transits"].astype(np.int32)
m = np.isfinite(sigma) & (nb_transits >= 3) & (conf > 0.75)
pval = data["rv_pval"][m]

fig, ax1 = plt.subplots(1, 1, figsize=(5, 3))
_, bins, _ = ax1.hist(
    pval, 100, color="k", histtype="step", weights=1e-5 + np.zeros(len(pval))
)
expect = 1e-5 * len(pval) / (len(bins) - 1)
ax1.plot([0, 0, 1, 1], [0, expect, expect, 0], ":C1")
ax1.set_xlabel("$p$-value")
ax1.set_ylabel(r"number of targets [$\times 10^{5}$]")
ax1.set_yscale("log")

# ax2.hist(
#     pval,
#     100,
#     range=(0, 5 * THRESHOLD),
#     color="k",
#     histtype="step",
#     weights=1e-5 + np.zeros(len(pval)),
# )
# ax2.fill_between(
#     [0, THRESHOLD],
#     0,
#     1,
#     color="C1",
#     alpha=0.3,
#     transform=ax2.get_xaxis_transform(),
# )
# ax2.set_xlabel("p-value")
# ax2.set_ylabel(r"number of targets [$\times 10^{5}$]")
# ax2.set_yscale("log")

fig.savefig("p_value_dist.pdf", bbox_inches="tight")
