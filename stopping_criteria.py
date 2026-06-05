import numpy as np

def primal_eps(p, x, z, eps_abs=1e-3, eps_rel=1e-3):
    """Primal stopping criterion for ADMM."""
    return np.sqrt(p) * eps_abs + eps_rel * np.max([np.linalg.norm(x), np.linalg.norm(z)])

def dual_eps(n, u, rho, eps_abs=1e-3, eps_rel=1e-3):
    """Dual stopping criterion for ADMM."""
    return np.sqrt(n) * eps_abs + eps_rel * np.linalg.norm(rho * u)