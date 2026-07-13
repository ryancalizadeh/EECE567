import numpy as np
from typing import Callable, Dict
import casadi as ca

def admm(f: Callable,
		 g: Callable,
		 z0: np.ndarray,
		 rho=lambda i, prev, r, s: 2.0,
		 threshold=1e-3,
		 max_iterations=1000,
		 callback=None):
	"""
	Minimizes a constrained optimization problem using the Alternating Direction Method of Multipliers (ADMM).

	Parameters
	----------
	f : Proxable
		The (possibly constrained) objective function to be minimized.
	g : Proxable
		The projection operator representing the constraints.
	z0 : Trajectory
		The initial guess for the solution.
	callback : callable, optional 
		Called as callback(iteration, x, z, u) at the end of each iteration.
	"""

	# Initialize x0, z0, mu0
	zs: list[np.ndarray] = [z0.copy()]
	xs: list[np.ndarray] = [np.zeros_like(z0)]
	us: list[np.ndarray] = [np.zeros_like(z0)]

	rs = [np.linalg.norm(xs[-1] - zs[-1])]
	ss = [np.linalg.norm(xs[-1] - zs[-1])]
	rhos = [2.0]

	for iteration in range(max_iterations-1):
		new_rho = rho(iteration, rhos[-1], rs[-1], ss[-1])
		rhos.append(new_rho)
		# Rescale u when rho changes to keep λ = rho*u continuous
		if new_rho != rhos[-2]:
			us[-1] = us[-1] * (rhos[-2] / new_rho)
			
		xs.append(f.prox(zs[-1] - us[-1], rhos[-1]))
		zs.append(g.prox(xs[-1] + us[-1], rhos[-1]))
		us.append(us[-1] + (xs[-1] - zs[-1]))

 
		rs.append(np.linalg.norm(xs[-1] - zs[-1]))
		ss.append(rhos[-1] * np.linalg.norm(zs[-1] - zs[-2]))

		if callback is not None:
			callback(iteration, xs[-1], zs[-1], us[-1], rs[-1], ss[-1])

		if rs[-1] < threshold and ss[-1] < threshold:
			break

	return xs[-1]

def centralized(config: Dict):
    Ybus_re = np.real(config["Ybus"])
    Ybus_im = np.imag(config["Ybus"])

    # ------------------------------------------------------------------ #
    # Decision variables
    # ------------------------------------------------------------------ #
    V_re = ca.SX.sym("V_re", config["n_buses"]) # type: ignore
    V_im = ca.SX.sym("V_im", config["n_buses"]) # type: ignore
    I_re = ca.SX.sym("I_re", config["n_buses"]) # type: ignore
    I_im = ca.SX.sym("I_im", config["n_buses"]) # type: ignore
    P = ca.SX.sym("P", config["n_buses"]) # type: ignore
    Q = ca.SX.sym("Q", config["n_buses"]) # type: ignore

    constraints = []
	