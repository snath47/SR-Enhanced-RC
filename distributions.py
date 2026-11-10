"""Coupling weight distributions: homogeneous, 2GMM, shifted exponential."""

import numpy as np

G_MIN = 0.005
G_MAX = 0.05
SIGMA_2GMM = 0.002


def make_weights(dist, E, seed=7777, **kwargs):
    """Generate coupling weights for E edges.

    dist: 'hom', '2gmm', 'exp'
    Returns (gv, egw) where gv = raw weights, egw = normalised.
    """
    rng = np.random.default_rng(seed)

    if dist == "hom":
        g_mean = kwargs["g_mean"]
        gv = np.full(E, g_mean, np.float32)

    elif dist == "2gmm":
        g_mean = kwargs["g_mean"]
        dmu = kwargs["dmu"]
        h = E // 2
        mu1 = g_mean - dmu / 2
        mu2 = g_mean + dmu / 2
        gv1 = rng.normal(loc=mu1, scale=SIGMA_2GMM, size=h).astype(np.float32)
        gv2 = rng.normal(loc=mu2, scale=SIGMA_2GMM, size=E - h).astype(np.float32)
        gv = np.clip(np.concatenate([gv1, gv2]), G_MIN, G_MAX)

    elif dist == "exp":
        beta = kwargs["beta"]
        x0 = kwargs.get("x0", 0.0)
        gv = np.clip(x0 + rng.exponential(scale=beta, size=E).astype(np.float32), G_MIN, G_MAX)

    else:
        raise ValueError(f"Unknown distribution: {dist}")

    return gv
