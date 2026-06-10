from Proxable import Proxable
from Trajectory import Trajectory
import numpy as np
from typing import Callable
from scipy.optimize import minimize_scalar

class ConstPowerLoad(Proxable):
	"""
	A class implementing projections onto the behaviour of a constant power load
	"""

	def __init__(self, S: Callable[[float], complex], max_iter=20, tol=1e-5):
		self.S = S
		self.max_iter = max_iter
		self.tol = tol


	def prox(self, trajectory: Trajectory, rho=1.0) -> Trajectory:
		ret = trajectory.copy()
		for n in range(ret.N):
			V0 = ret.w["voltage"][:, [n]]
			I0 = ret.w["current"][:, [n]]

			s = self.S(n * ret.dt)

			# Precompute constants for the objective function
			A = (np.abs(V0)**2).item()
			B = (np.abs(I0)**2).item()
			C = (2 * np.real(np.conj(s) * V0 * np.conj(I0))).item()
			abs_s = np.abs(s)

			# 1D Objective function based on the magnitude r = |i|.
			# This mirrors the scalar reduction approach in the paper.
			def objective(r):
				# W_mag represents the magnitude of the optimally phase-aligned vector
				W_mag = np.sqrt(B * r**2 + A * (abs_s**2) / r**2 + C)
				# The simplified distance function
				return r**2 + (abs_s**2) / r**2 - 2 * W_mag

			# Find the optimal magnitude r > 0.
			res = minimize_scalar(objective, bounds=(1e-6, 1e6), method='bounded')
			r_opt = res.x # type: ignore

			# Compute the optimal complex phase alignment vector W
			W = (np.conj(s) / r_opt) * V0 + r_opt * I0
			W_mag = np.abs(W)

			# Handle the edge case where W is perfectly zero
			if W_mag.item() < 1e-12:
				# keep the phase as an array matching V0's shape to avoid scalars
				phase = np.ones_like(V0, dtype=complex)
			else:
				phase = W / W_mag

			# Reconstruct the final projected complex values (ensure arrays)
			final_i = (r_opt * phase).reshape(V0.shape)
			final_v = ((s / r_opt) * phase).reshape(V0.shape)

			ret.w["voltage"][:, n] = final_v.ravel()
			ret.w["current"][:, n] = final_i.ravel()
			# Store power per channel (broadcast scalar s to channel vector if needed)
			ret.w["power"][:, n] = np.full(ret.w["power"][:, n].shape, s)

		return ret


def test_const_load_projection():
	v = 2.5 + 1.5j
	i = 0.9 - 0.5j
	s = 3.1 - 1.4j

	load = ConstPowerLoad(lambda t: s)
	traj = Trajectory(0.03, 0.01, {"voltage": 1, "current": 1, "power": 1}, dtype=np.complex64)
	traj.set_constant(["voltage"], [v])
	traj.set_constant(["current"], [i])
	projected_traj = load.prox(traj)
	print("Projected Voltage:", projected_traj.w['voltage'])
	print("Projected Current:", projected_traj.w['current'])
	print("Power at each time step:", projected_traj.w['voltage'] * projected_traj.w['current'].conjugate())

	# Check optimality
	vp = projected_traj.w["voltage"][0, 0]
	ip = projected_traj.w["current"][0, 0]
	print("Stationary Residual", np.abs(np.abs(vp)**2 - v*np.conj(vp) - (np.abs(ip)**2 - ip*np.conj(i))))

if __name__ == "__main__":
	test_const_load_projection()