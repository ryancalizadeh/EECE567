from Solvable import Solvable
from Trajectory import Trajectory
import cvxpy as cp
import numpy as np

class Objective(Solvable):
	"""
	A class reprsenting the minimization of generation cost subject to operational constraints and linear network behaviour
	"""

	def __init__(self, Ybus, gen_costs, P_min, P_max, V_max, t: Trajectory, rho=1.0):
		self.Ybus = Ybus
		self.gen_costs = gen_costs
		self.N = t.N
		self.g = len(gen_costs)
		self.n_buses = Ybus.shape[0]

		self.rho = rho

		P_min = np.array(P_min).reshape(-1, 1)
		P_max = np.array(P_max).reshape(-1, 1)
		V_max = np.array(V_max).reshape(-1, 1)


		self.V = cp.Variable((Ybus.shape[0], self.N), complex=True)
		self.I = cp.Variable((Ybus.shape[0], self.N), complex=True)
		self.S = cp.Variable((self.g, self.N), complex=True)
		self.Pc_var = cp.Variable((self.g, self.N))
		self.Vw = cp.Parameter(self.V.shape, complex=True)
		self.Iw = cp.Parameter(self.I.shape, complex=True)
		self.Sw = cp.Parameter(self.S.shape, complex=True)
		self.Pcw = cp.Parameter((self.g, self.N))

		self.x = cp.vstack([self.V, self.I, self.S])
		self.w = cp.vstack([self.Vw, self.Iw, self.Sw])

		# TODO Reevaluate this since it might be wrong
		# Self.cost = sum over each generator and each timestep of real(S_i)^2 * gen_costs[i]
		self.cost = cp.sum([cp.quad_over_lin(cp.real(self.S[i, :]), 1/self.gen_costs[i]) for i in range(self.g)])

		# Self.penalty = rho * ||x-w||_2^2, recalling that these are complex numbers
		self.penalty = self.rho * cp.sum_squares(self.x - self.w)
		self.Pc_penalty = self.rho * cp.sum_squares(self.Pc_var - self.Pcw)

		self.constraints = [self.Ybus @ self.V == self.I]
		self.constraints.append(cp.real(self.S) >= P_min)
		self.constraints.append(cp.real(self.S) <= P_max)
		self.constraints.append(cp.abs(self.V) <= V_max)
		self.constraints.append(self.Pc_var >= P_min)
		self.constraints.append(self.Pc_var <= P_max)

		# TODO Add line current limits

		self.problem = cp.Problem(cp.Minimize(self.cost + self.penalty + self.Pc_penalty), self.constraints)
		


	def solve(self, t: Trajectory) -> Trajectory:
		self.Vw.value = t.get_var_names(["voltage"])
		self.Iw.value = t.get_var_names(["current"])
		self.Sw.value = t.get_var_names(["power"])[:self.g, :]
		self.Pcw.value = np.real(t.get_var_names(["Pc"]))

		self.problem.solve()

		# Extract solution and return a trajectory
		ret = t.copy()

		if self.V.value is not None:
			ret.set_var_names(["voltage"], self.V.value)
		if self.I.value is not None:
			ret.set_var_names(["current"], self.I.value)
		if self.S.value is not None:
			ret.w["power"][:self.g, :] = self.S.value
		if self.Pc_var.value is not None:
			ret.w["Pc"][:self.g, :] = self.Pc_var.value

		if self.problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
			print(f"Optimization failed with status {self.problem.status}")

		return ret
