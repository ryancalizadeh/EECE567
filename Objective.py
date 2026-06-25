from Proxable import Proxable
from Trajectory import Trajectory
import cvxpy as cp
import numpy as np

class Objective(Proxable):
	"""
	A class reprsenting the minimization of generation cost subject to operational constraints and linear network behaviour
	"""

	def __init__(self, Ybus, gen_costs, P_min, P_max, V_max, t: Trajectory, omega_s: float = 2*3.141592653589793*60, omega_band: float = 0.08, rho=2.0):
		self.Ybus = Ybus
		self.gen_costs = gen_costs
		self.N = t.N
		self.g = len(gen_costs)
		self.n_buses = Ybus.shape[0]
		self.P_min = np.array(P_min).reshape(-1, 1)
		self.P_max = np.array(P_max).reshape(-1, 1)
		self.V_max = np.array(V_max).reshape(-1, 1)
		self.omega_band = omega_band
		self.omega_s = omega_s

		self.V = cp.Variable((Ybus.shape[0], self.N), complex=True)
		self.I = cp.Variable((Ybus.shape[0], self.N), complex=True)
		self.S = cp.Variable((self.g, self.N), complex=True)
		self.Pc_var = cp.Variable((self.g, self.N))
		self.omega = cp.Variable((self.g, self.N))
		self.Tm = cp.Variable((self.g, self.N))
		self.Vw = cp.Parameter(self.V.shape, complex=True)
		self.Iw = cp.Parameter(self.I.shape, complex=True)
		self.Sw = cp.Parameter(self.S.shape, complex=True)
		self.Pcw = cp.Parameter((self.g, self.N))
		self.omegaw = cp.Parameter((self.g, self.N))
		self.Tmw = cp.Parameter((self.g, self.N))

		self.rho = rho

		self.x = cp.vstack([self.V, self.I, self.S])
		self.w = cp.vstack([self.Vw, self.Iw, self.Sw])

		# TODO Reevaluate this since it might be wrong
		# Self.cost = sum over each generator and each timestep of real(S_i)^2 * gen_costs[i]
		# self.cost = cp.sum([cp.quad_over_lin(cp.real(self.S[i, :]), 1/self.gen_costs[i]) for i in range(self.g)])
		self.cost = cp.sum([cp.quad_over_lin(self.Tm[i, :], 1/self.gen_costs[i]) for i in range(self.g)])

		# Self.penalty = rho/2 * ||x-w||_2^2, recalling that these are complex numbers
		self.penalty = self.rho/2 * cp.sum_squares(self.x - self.w)
		self.Pc_penalty = self.rho/2 * cp.sum_squares(self.Pc_var - self.Pcw)
		self.omega_penalty = self.rho/2 * cp.sum_squares(self.omega - self.omegaw)
		self.Tm_penalty = self.rho/2 * cp.sum_squares(self.Tm - self.Tmw)

		# TODO: Verify that optimization problem is the same as DAOPF formulation

		omega_s_arr = np.full((self.g, 1), self.omega_s)

		self.constraints = [self.Ybus @ self.V == self.I]
		self.constraints.append(cp.real(self.S) >= self.P_min)
		self.constraints.append(cp.real(self.S) <= self.P_max)
		self.constraints.append(cp.abs(self.V) <= self.V_max)
		self.constraints.append(self.Pc_var >= self.P_min)
		self.constraints.append(self.Pc_var <= self.P_max)
		self.constraints.append(self.omega >= omega_s_arr - self.omega_band)
		self.constraints.append(self.omega <= omega_s_arr + self.omega_band)

		# TODO Add line current limits

		self.problem = cp.Problem(cp.Minimize(self.cost + self.penalty + self.Pc_penalty + self.omega_penalty + self.Tm_penalty), self.constraints)

	def create_problem(self, rho):
		self.rho = rho

		# TODO Reevaluate this since it might be wrong
		# Self.cost = sum over each generator and each timestep of real(S_i)^2 * gen_costs[i]
		self.cost = cp.sum([cp.quad_over_lin(cp.real(self.S[i, :]), 1/self.gen_costs[i]) for i in range(self.g)])
		# self.cost = cp.sum([cp.quad_over_lin(self.Tm[i, :], 1/self.gen_costs[i]) for i in range(self.g)])

		# Self.penalty = rho/2 * ||x-w||_2^2, recalling that these are complex numbers
		self.penalty = self.rho/2 * cp.sum_squares(self.x - self.w)
		self.Pc_penalty = self.rho/2 * cp.sum_squares(self.Pc_var - self.Pcw)
		self.omega_penalty = self.rho/2 * cp.sum_squares(self.omega - self.omegaw)

		# TODO: Verify that optimization problem is the same as DAOPF formulation

		omega_s_arr = np.full((self.g, 1), self.omega_s)

		self.constraints = [self.Ybus @ self.V == self.I]
		self.constraints.append(cp.real(self.S) >= self.P_min)
		self.constraints.append(cp.real(self.S) <= self.P_max)
		self.constraints.append(cp.abs(self.V) <= self.V_max)
		self.constraints.append(self.Pc_var >= self.P_min)
		self.constraints.append(self.Pc_var <= self.P_max)
		self.constraints.append(self.omega >= omega_s_arr - self.omega_band)
		self.constraints.append(self.omega <= omega_s_arr + self.omega_band)

		# TODO Add line current limits

		self.problem = cp.Problem(cp.Minimize(self.cost + self.penalty + self.Pc_penalty + self.omega_penalty + self.Tm_penalty), self.constraints)
		

	def prox(self, trajectory: Trajectory, rho: float = 1.0) -> Trajectory:
		self.Vw.value = trajectory.get_var_names(["voltage"])
		self.Iw.value = trajectory.get_var_names(["current"])
		self.Sw.value = trajectory.get_var_names(["power"])[:self.g, :]
		self.Pcw.value = np.real(trajectory.get_var_names(["Pc"]))
		self.omegaw.value = np.real(trajectory.get_var_names(["omega"]))
		self.Tmw.value = trajectory.get_var_names(["Tm"])[:self.g, :]

		#self.rho.value = rho

		self.problem.solve()

		# Extract solution and return a trajectory
		ret = trajectory.copy()

		if self.V.value is not None:
			ret.set_var_names(["voltage"], self.V.value)
		if self.I.value is not None:
			ret.set_var_names(["current"], self.I.value)
		if self.S.value is not None:
			ret.w["power"][:self.g, :] = self.S.value
		if self.Pc_var.value is not None:
			ret.w["Pc"][:self.g, :] = self.Pc_var.value
		if self.omega.value is not None:
			ret.w["omega"][:self.g, :] = self.omega.value
		if self.Tm.value is not None:
			ret.w["Tm"][:self.g, :] = self.Tm.value


		if self.problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
			print(f"Optimization failed with status {self.problem.status}")

		return ret
