from Projectable import Projectable
from Trajectory import Trajectory
import numpy as np
from typing import Callable
from scipy.optimize import minimize_scalar

class ConstPowerLoad(Projectable):
	"""
	A class implementing projections onto the behaviour of a constant power load
	"""

	def __init__(self, S: Callable[[float], complex], max_iter=20, tol=1e-5):
		self.S = S
		self.max_iter = max_iter
		self.tol = tol


	def project(self, trajectory: Trajectory) -> Trajectory:
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
				phase = 1.0 + 0j
			else:
				phase = W / W_mag

			# Reconstruct the final projected complex values
			final_i = r_opt * phase
			final_v = (s / r_opt) * phase

			ret.w["voltage"][:, n] = final_v.ravel()
			ret.w["current"][:, n] = final_i.ravel()
			ret.w["power"][:, n] = s

		return ret