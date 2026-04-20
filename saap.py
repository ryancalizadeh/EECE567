import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Callable
import casadi as ca
from scipy.optimize import minimize_scalar
import cvxpy as cp


class Projectable(ABC):
	"""
	An abstract base class representing a projectable set.

	Methods
	-------
	project(trajectory: Trajectory) -> Trajectory
		Projects the given trajectory onto the set.
	"""

	@abstractmethod
	def project(self, trajectory: Trajectory) -> Trajectory:
		pass

class Generator(Projectable):
	"""
	A class implementing projections onto a single classical generator + fast governor behaviour
	"""

	def __init__(self, E, Zd, H, D, Tsv, R, delta0, Pc, V0, I0, S0, max_iter=20, tol=1e-5):
		self.E = E
		self.Zd = Zd
		self.H = H
		self.D = D
		self.Tsv = Tsv
		self.R = R
		self.delta0 = delta0
		self.Pc = Pc
		self.ws = 2*np.pi*60
		self.max_iter = max_iter
		self.tol = tol
		self.V0 = V0
		self.I0 = I0
		self.S0 = S0

	def project(self, trajectory: Trajectory) -> Trajectory:
		"""
		Projects the given trajectory onto the generator's behaviour. In particular, it solves

		min ||w - trajectory||_2^2
		s.t. w satisfies the generator's DAE constraints, which are discretized using the trapezoidal rule.
		
		Uses CasADI and IPOPT to solve the resulting nonlinear program, and uses the previous solution as a warm start for the next time step to speed up convergence.
		
		Parameters
		----------
		trajectory : Trajectory
			The trajectory to project.
		
		Returns
		-------
		Trajectory
			The projected trajectory.
		"""
		
		ret = trajectory.copy()
		N = trajectory.N
		dt = trajectory.dt

		# Total number of variables: delta, omega, Tm, V_re, V_im, I_re, I_im, P, Q
		n_vars = 9

		# Extract measurement data
		V_traj = trajectory.get_var_names(["voltage"])
		I_traj = trajectory.get_var_names(["current"])
		delta_traj = trajectory.get_var_names(["delta"])
		omega_traj = trajectory.get_var_names(["omega"])
		Tm_traj = trajectory.get_var_names(["Tm"])
		S_traj = trajectory.get_var_names(["power"])

		V_traj_re = np.real(V_traj)
		V_traj_im = np.imag(V_traj)
		I_traj_re = np.real(I_traj)
		I_traj_im = np.imag(I_traj)
		delta_traj = np.real(delta_traj)
		omega_traj = np.real(omega_traj)
		Tm_traj = np.real(Tm_traj)
		P_traj = np.real(S_traj)
		Q_traj = np.imag(S_traj)

		# Create optimization variables (as 2D matrix for easier indexing)
		x_matrix = ca.MX.sym('x', (n_vars, N))  # type: ignore
		# Flatten to vector for nlpsol (which requires a dense vector)
		x = ca.reshape(x_matrix, -1, 1)
		
		delta = x_matrix[0, :]
		omega = x_matrix[1, :]
		Tm = x_matrix[2, :]
		V_re = x_matrix[3, :]
		V_im = x_matrix[4, :]
		I_re = x_matrix[5, :]
		I_im = x_matrix[6, :]
		P = x_matrix[7, :]
		Q = x_matrix[8, :]

		# Build constraints vector
		constraints_list = []
		
		# Initial conditions
		constraints_list.append(delta[0])
		constraints_list.append(omega[0] - self.ws)
		constraints_list.append(Tm[0] - self.Pc)
		constraints_list.append(V_re[0] - self.V0.real)
		constraints_list.append(V_im[0] - self.V0.imag)
		constraints_list.append(I_re[0] - self.I0.real)
		constraints_list.append(I_im[0] - self.I0.imag)
		constraints_list.append(P[0] - self.S0.real)
		constraints_list.append(Q[0] - self.S0.imag)

		# # Add temporary constraint to fix variables for the first 0.5 seconds
		# for k in range(min(int(0.5/dt), N)):
		# 	constraints_list.append(V_re[k] - V_traj_re[0, k])
		# 	constraints_list.append(V_im[k] - V_traj_im[0, k])
		# 	constraints_list.append(I_re[k] - I_traj_re[0, k])
		# 	constraints_list.append(I_im[k] - I_traj_im[0, k])
		# 	constraints_list.append(omega[k] - self.ws)
		# 	constraints_list.append(Tm[k] - self.Pc)
		# 	constraints_list.append(delta[k+1] - delta[k])
		
		# Dynamics constraints (trapezoidal rule)
		for k in range(N - 1):
			# delta_dot = omega - ws
			f_delta_k = omega[k] - self.ws
			f_delta_k1 = omega[k+1] - self.ws
			constraints_list.append(delta[k+1] - delta[k] - (dt/2) * (f_delta_k + f_delta_k1))
			
			# omega_dot
			E_k_re = self.E * ca.cos(delta[k] + self.delta0)
			E_k_im = self.E * ca.sin(delta[k] + self.delta0)
			Pe_k = E_k_re * I_re[k] + E_k_im * I_im[k]
			
			E_k1_re = self.E * ca.cos(delta[k+1] + self.delta0)
			E_k1_im = self.E * ca.sin(delta[k+1] + self.delta0)
			Pe_k1 = E_k1_re * I_re[k+1] + E_k1_im * I_im[k+1]
			
			f_omega_k = (Tm[k] - Pe_k - self.D*(omega[k]/self.ws - 1)) * self.ws/(2*self.H)
			f_omega_k1 = (Tm[k+1] - Pe_k1 - self.D*(omega[k+1]/self.ws - 1)) * self.ws/(2*self.H)
			constraints_list.append(omega[k+1] - omega[k] - (dt/2) * (f_omega_k + f_omega_k1))
			
			# Tm_dot
			f_Tm_k = (-Tm[k] + self.Pc - 1/self.R * (omega[k]/self.ws - 1)) / self.Tsv
			f_Tm_k1 = (-Tm[k+1] + self.Pc - 1/self.R * (omega[k+1]/self.ws - 1)) / self.Tsv
			constraints_list.append(Tm[k+1] - Tm[k] - (dt/2) * (f_Tm_k + f_Tm_k1))
		
		# Algebraic constraints
		for k in range(N):
			E_k_re = self.E * ca.cos(delta[k] + self.delta0)
			E_k_im = self.E * ca.sin(delta[k] + self.delta0)
			# E - Zd*I - V = 0
			alg_re = E_k_re - self.Zd.real * I_re[k] + self.Zd.imag * I_im[k] - V_re[k]
			alg_im = E_k_im - self.Zd.real * I_im[k] - self.Zd.imag * I_re[k] - V_im[k]
			constraints_list.append(alg_re)
			constraints_list.append(alg_im)

			# Add electrical power constraints
			P_k = V_re[k] * I_re[k] + V_im[k] * I_im[k]
			Q_k = V_im[k] * I_re[k] - V_re[k] * I_im[k]

			constraints_list.append(P[k] - P_k)
			constraints_list.append(Q[k] - Q_k)

		# Terminal condition: enforce steady-state at final timestep
		k = N - 1
		f_delta_N = omega[k] - omega[k-1]
		E_k_re = self.E * ca.cos(delta[k] + self.delta0)
		E_k_im = self.E * ca.sin(delta[k] + self.delta0)
		Pe_k = E_k_re * I_re[k] + E_k_im * I_im[k]
		f_omega_N = (Tm[k] - Pe_k - self.D*(omega[k]/self.ws - 1)) * self.ws/(2*self.H)
		f_Tm_N = (-Tm[k] + self.Pc - 1/self.R * (omega[k]/self.ws - 1)) / self.Tsv

		# constraints_list.append(f_delta_N)   # omega[N-1] == ws
		# constraints_list.append(f_omega_N)   # Tm[N-1] == Pe at final step
		# constraints_list.append(f_Tm_N)      # Tm[N-1] at steady state

		
		constraints_vec = ca.vertcat(*constraints_list)

		lbg = np.zeros(constraints_vec.shape[0])
		ubg = np.zeros(constraints_vec.shape[0])

		# Objective function: minimize ||w - trajectory||_2^2
		obj = ca.sumsqr(V_re - V_traj_re) + ca.sumsqr(V_im - V_traj_im) + ca.sumsqr(I_re - I_traj_re) + ca.sumsqr(I_im - I_traj_im) + ca.sumsqr(omega - omega_traj) + ca.sumsqr(Tm - Tm_traj) + ca.sumsqr(delta - delta_traj) + ca.sumsqr(P - P_traj) + ca.sumsqr(Q - Q_traj)


		opts = {
			'ipopt.print_level': 5, # 0 for silent, 5 for detailed output
			'ipopt.tol': 1e-3,
			'ipopt.max_iter': 500,
			'ipopt.acceptable_tol': 1e-5,
			'ipopt.acceptable_iter': 10,
			'ipopt.mu_strategy': 'adaptive',
			'ipopt.nlp_scaling_method': 'gradient-based',
			'ipopt.alpha_for_y': 'min-dual-infeas',
			'ipopt.recalc_y': 'yes'
		}

		# Create NLP solver (requires dense vector x)
		nlp_solver = ca.nlpsol('nlp_solver', 'ipopt', {'f': obj, 'g': constraints_vec, 'x': x}, opts)

		# Initial guess - improved initialization
		# Start with measurement data and gradually adjust
		delta_guess = np.zeros_like(V_traj_re[0, :])  # Start with zero angle deviations
		omega_guess = np.ones((N,)) * self.ws
		Tm_guess = np.ones((N,)) * self.Pc
		V_re_guess = V_traj_re.flatten()
		V_im_guess = V_traj_im.flatten()
		I_re_guess = I_traj_re.flatten()
		I_im_guess = I_traj_im.flatten()
		P_guess = P_traj.flatten()
		Q_guess = Q_traj.flatten()
		
		x_init = np.column_stack([delta_guess, omega_guess, Tm_guess, V_re_guess, V_im_guess, I_re_guess, I_im_guess, P_guess, Q_guess])
		initial_guess = x_init.flatten()

		lbx = np.full((N, n_vars), -np.inf)
		ubx = np.full((N, n_vars), np.inf)
		lbx[:, 1] = self.ws - 0.08
		ubx[:, 1] = self.ws + 0.08

		# Solve NLP
		print("Solving NLP...")
		sol = nlp_solver(x0=initial_guess, lbg=lbg, ubg=ubg, lbx=lbx.flatten(), ubx=ubx.flatten())

		# Extract solution and reshape back to 2D form
		sol_x_flat = sol['x'].full().flatten()
		sol_x = sol_x_flat.reshape((N, n_vars)).T

		sol_delta = sol_x[0, :].reshape(1, -1)
		sol_omega = sol_x[1, :].reshape(1, -1)
		sol_Tm = sol_x[2, :].reshape(1, -1)
		sol_V_re = sol_x[3, :]
		sol_V_im = sol_x[4, :]
		sol_I_re = sol_x[5, :]
		sol_I_im = sol_x[6, :]
		sol_V = (sol_V_re + 1j*sol_V_im).reshape(1, -1)
		sol_I = (sol_I_re + 1j*sol_I_im).reshape(1, -1)
		sol_P = sol_x[7, :].reshape(1, -1)
		sol_Q = sol_x[8, :].reshape(1, -1)
		sol_S = (sol_P + 1j * sol_Q)

		# # Plot omega
		# fig, axs = plt.subplots(1, 1, figsize=(10, 5))
		# axs.plot(np.arange(N)*dt, sol_omega[0, :], label='Projected Omega')
		# axs.axhline(self.ws, color='r', linestyle='--', label='Synchronous Speed')
		# axs.set_title('Projected Rotor Speed (Omega)')
		# axs.set_xlabel('Time (s)')
		# axs.set_ylabel('Omega (rad/s)')
		# axs.set_ylim(self.ws - 0.08, self.ws + 0.08)
		# axs.legend()
		# plt.grid()
		# plt.show()

		# # Plot voltage
		# fig, axs = plt.subplots(1, 1, figsize=(10, 5))
		# axs.plot(np.arange(N)*dt, sol_V_re, label='Projected V_re')
		# axs.plot(np.arange(N)*dt, sol_V_im, label='Projected V_im')
		# axs.set_title('Projected Voltage')
		# axs.set_xlabel('Time (s)')
		# axs.set_ylabel('Voltage (V)')
		# axs.legend()
		# plt.grid()
		# plt.show()

		ret.set_var_names(["voltage"], sol_V)
		ret.set_var_names(["current"], sol_I)
		ret.set_var_names(["delta"], sol_delta)
		ret.set_var_names(["omega"], sol_omega)
		ret.set_var_names(["Tm"], sol_Tm)
		ret.set_var_names(["power"], sol_S)

		return ret
	
	def f(self, x):
		delta = x[0]
		omega = x[1]
		Tm = x[2]
		V_re = x[3]
		V_im = x[4]
		I_re = x[5]
		I_im = x[6]

		E_re = self.E * ca.cos(delta + self.delta0)
		E_im = self.E * ca.sin(delta + self.delta0)
		Pe = E_re * I_re + E_im * I_im

		delta_dot = omega - self.ws
		omega_dot = (Tm - Pe - self.D*(omega/self.ws - 1)) * self.ws/(2*self.H)
		Tm_dot = (-Tm + self.Pc - 1/self.R * (omega/self.ws - 1)) / self.Tsv
		alg_re = E_re - self.Zd.real * I_re + self.Zd.imag * I_im - V_re
		alg_im = E_im - self.Zd.real * I_im - self.Zd.imag * I_re - V_im

		return ca.vertcat(delta_dot, omega_dot, Tm_dot, alg_re, alg_im)
	
	def ae(self, x):
		delta = x[0]
		omega = x[1]
		Tm = x[2]
		V_re = x[3]
		V_im = x[4]
		I_re = x[5]
		I_im = x[6]

		E_re = self.E * ca.cos(delta + self.delta0)
		E_im = self.E * ca.sin(delta + self.delta0)
		
		alg_re = E_re - self.Zd.real * I_re + self.Zd.imag * I_im - V_re
		alg_im = E_im - self.Zd.real * I_im - self.Zd.imag * I_re - V_im

		return ca.vertcat(alg_re, alg_im)

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

class BusBehaviours(Projectable):
	def __init__(self, gens: list[Generator], loads: list[ConstPowerLoad]):
		self.gens = gens
		self.loads = loads

	def project(self, trajectory: Trajectory) -> Trajectory:
		ret = trajectory.copy()
		for i, gen in enumerate(self.gens):
			print(f"Projecting onto generator {i+1}/{len(self.gens)}")
			gen_vars = ["voltage", "current", "delta", "omega", "Tm", "power"]
			t = Trajectory(trajectory.T, trajectory.dt, {v: 1 for v in gen_vars})
			for v in gen_vars:
				t.w[v] = trajectory.w[v][[i], :]
			projected_t = gen.project(t)
			for v in gen_vars:
				ret.w[v][[i], :] = projected_t.w[v]
		for i, load in enumerate(self.loads, start=len(self.gens)):
			print(f"Projecting onto load {i+1}/{len(self.loads)}")
			load_vars = ["voltage", "current", "power"]
			t = Trajectory(trajectory.T, trajectory.dt, {v: 1 for v in load_vars})
			for v in load_vars:
				t.w[v] = trajectory.w[v][[i], :]
			projected_t = load.project(t)
			for v in load_vars:
				ret.w[v][[i], :] = projected_t.w[v]
		return ret

class Solvable(ABC):
	"""
	An abstract base class representing an optimization problem parameterized by a trajectory, where the solution is a trajectory that satisfies certain constraints.

	Methods
	-------
	solve(trajectory: Trajectory) -> Trajectory
		Solves the optimization problem for the given trajectory.
	"""

	@abstractmethod
	def solve(self, t: Trajectory) -> Trajectory:
		pass

class Objective(Solvable):
	"""
	A class reprsenting the minimization of generation cost subject to operational constraints and linear network behaviour
	"""

	def __init__(self, Ybus, gen_costs, P_min, P_max, V_max, t: Trajectory, rho=1.0):
		self.Ybus = Ybus
		self.gen_costs = gen_costs
		self.N = t.N
		self.g = len(gen_costs)
		self.num_buses = Ybus.shape[0]

		self.rho = rho

		P_min = np.array(P_min).reshape(-1, 1)
		P_max = np.array(P_max).reshape(-1, 1)
		V_max = np.array(V_max).reshape(-1, 1)


		self.V = cp.Variable((Ybus.shape[0], self.N), complex=True)
		self.I = cp.Variable((Ybus.shape[0], self.N), complex=True)
		self.S = cp.Variable((self.g, self.N), complex=True)
		self.Vw = cp.Parameter(self.V.shape, complex=True)
		self.Iw = cp.Parameter(self.I.shape, complex=True)
		self.Sw = cp.Parameter(self.S.shape, complex=True)

		self.x = cp.vstack([self.V, self.I, self.S])
		self.w = cp.vstack([self.Vw, self.Iw, self.Sw])

		# TODO Reevaluate this since it might be wrong
		# Self.cost = sum over each generator and each timestep of real(S_i)^2 * gen_costs[i]
		self.cost = cp.sum([cp.quad_over_lin(cp.real(self.S[i, :]), 1/self.gen_costs[i]) for i in range(self.g)])

		# Self.penalty = rho * ||x-w||_2^2, recalling that these are complex numbers
		self.penalty = self.rho * cp.sum_squares(self.x - self.w)

		self.constraints = [self.Ybus @ self.V == self.I]
		self.constraints.append(cp.real(self.S) >= P_min)
		self.constraints.append(cp.real(self.S) <= P_max)
		self.constraints.append(cp.abs(self.V) <= V_max)

		# TODO Add line current limits

		self.problem = cp.Problem(cp.Minimize(self.cost + self.penalty), self.constraints)
		


	def solve(self, t: Trajectory) -> Trajectory:
		self.Vw.value = t.get_var_names(["voltage"])
		self.Iw.value = t.get_var_names(["current"])
		self.Sw.value = t.get_var_names(["power"])[:self.g, :]

		self.problem.solve()

		# Extract solution and return a trajectory
		ret = t.copy()

		if self.V.value is not None:
			ret.set_var_names(["voltage"], self.V.value)
		if self.I.value is not None:
			ret.set_var_names(["current"], self.I.value)
		if self.S.value is not None:
			ret.w["power"][:self.g, :] = self.S.value
		
		if self.problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
			print(f"Optimization failed with status {self.problem.status}")

		return ret


class Trajectory:
	"""
	A class representing a trajectory with time points and corresponding states.

	Parameters
	----------
	T : int
		Time horizon of the trajectory.
	vars: dict
		A dictionary mapping variable names to their dimensions.
	"""

	def __init__(self, T: float, dt: float, vars: dict):
		self.T = T
		self.dt = dt
		self.N = int(T / dt)
		self.vars = vars
		self.q = sum(vars.values())
		self.w = {key: np.zeros((size, self.N), dtype=np.complex64) for key, size in vars.items()}

	def get_var_names(self, var_names: list[str], idx=None) -> np.ndarray:
		"""
		Returns the submatrix corresponding to the given variable names.
		
		Parameters
		----------
		var_names : list[str]
			The names of the variables to extract.
		idx : list[int], optional
			The indices of the rows to extract for each variable. If None, extracts all rows.
		"""

		if not var_names:
			return np.array([], dtype=np.complex64).reshape(0, self.N)

		if idx is None:
			sub_trajectories = [self.w[var_name] for var_name in var_names]
		else:
			sub_trajectories = [self.w[var_name][idx, :] for var_name in var_names]

		return np.vstack(sub_trajectories)

	def set_var_names(self, var_names: list[str], values: np.ndarray, idx=None) -> None:
		"""
		Sets the sub-trajectories for the given variable names.
		
		Parameters
		----------
		var_names : list[str]
			The names of the variables to set.
		values : np.ndarray
			The values to set for the specified variables.
		idx : list[int], optional
			The indices of the rows to set for each variable. If None, sets all rows.
		"""

		if not var_names:
			return

		start_row = 0
		for var_name in var_names:
			size = self.vars[var_name]
			if idx is None:
				self.w[var_name] = values[start_row:start_row+size, :]
			else:
				self.w[var_name][idx, :] = values[start_row:start_row+size, :]
			start_row += size

	def set_constant(self, var_names: list[str], value, idx=None) -> None:
		"""
		Sets the specified variables to a constant value.

		Parameters
		----------
		var_name : list[str]
			The names of the variables to set.
		value : scalar or np.ndarray
			The constant value to set the variables to.
		idx : list[int], optional
			The indices of the rows to set. If None, sets all rows.
		"""
		for var_name in var_names:
			# Convert value to numpy array and reshape if needed for broadcasting
			val = np.asarray(value)
			if val.ndim == 1:
				val = val.reshape(-1, 1)  # Shape (n,) -> (n, 1) for proper broadcasting
			
			if idx is None:
				self.w[var_name][:, :] = val
			else:
				self.w[var_name][idx, :] = val

	def __add__(self, other):
		if not isinstance(other, Trajectory):
			return NotImplemented
		if self.N != other.N or self.vars != other.vars:
			raise ValueError("Trajectories must have the same dimensions and variables for addition.")
		result = Trajectory(self.T, self.dt, self.vars)
		for var_name in self.vars:
			result.w[var_name] = self.w[var_name] + other.w[var_name]
		return result
	
	def __sub__(self, other):
		if not isinstance(other, Trajectory):
			return NotImplemented
		if self.N != other.N or self.vars != other.vars:
			raise ValueError("Trajectories must have the same dimensions and variables for subtraction.")
		result = Trajectory(self.T, self.dt, self.vars)
		for var_name in self.vars:
			result.w[var_name] = self.w[var_name] - other.w[var_name]
		return result
	
	def __mul__(self, scalar):
		result = Trajectory(self.T, self.dt, self.vars)
		for var_name in self.vars:
			result.w[var_name] = self.w[var_name] * scalar
		return result

	def __rmul__(self, scalar):
		return self.__mul__(scalar)

	def copy(self):
		"""
		Returns a deep copy of the trajectory.
		
		Returns
		-------
		Trajectory
			A new trajectory with the same dimensions and data.
		"""
		result = Trajectory(self.T, self.dt, self.vars)
		result.w = {key: value.copy() for key, value in self.w.items()}
		return result
	
	def norm(self):
		# Concatenate all variable trajectories into a single array and compute the norm
		all_vars = np.vstack(list(self.w.values()))
		return np.linalg.norm(all_vars)

def admm(f: Solvable, g: Projectable, z0: Trajectory, eta=lambda iteration: 1.0, threshold=1e-3, max_iterations=100, callback=None):
	"""
	Minimizes a constrained optimization problem using the Alternating Direction Method of Multipliers (ADMM).

	Parameters
	----------
	f : Solvable
		The (possibly constrained) objective function to be minimized.
	g : Projectable
		The projection operator representing the constraints.
	z0 : Trajectory
		The initial guess for the solution.
	callback : callable, optional
		Called as callback(iteration, x, z, u) at the end of each iteration.
	"""

	# Initialize x0, z0, mu0
	zs: list[Trajectory] = [z0.copy()]
	xs: list[Trajectory] = [Trajectory(z0.T, z0.dt, z0.vars)]
	us: list[Trajectory] = [Trajectory(z0.T, z0.dt, z0.vars)]

	for iteration in range(max_iterations-1):
		xs.append(f.solve(zs[-1] - us[-1]))
		zs.append(g.project(xs[-1] + us[-1]))
		us.append(us[-1] + eta(iteration) * (xs[-1] - zs[-1]))

		if callback is not None:
			callback(iteration, xs[-1], zs[-1], us[-1])

		if (xs[-1] - zs[-1]).norm() < threshold:
			break

	return xs[-1]

def test_const_load_projection():
	load = ConstPowerLoad(lambda t: 2.1 + 0.5j)
	traj = Trajectory(0.03, 0.01, {"voltage": 1, "current": 1, "power": 1})
	traj.set_constant(["voltage"], [1.5 + 1j])
	traj.set_constant(["current"], [2.2 - 0.3j])
	projected_traj = load.project(traj)
	print("Projected Voltage:", projected_traj.w['voltage'])
	print("Projected Current:", projected_traj.w['current'])
	print("Power at each time step:", projected_traj.w['voltage'] * projected_traj.w['current'].conjugate())

def test_get_set_var_names():
	traj = Trajectory(0.03, 0.01, {"voltage": 2, "current": 1})
	data = np.array([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]], dtype=np.complex64)
	traj.set_var_names(["voltage"], data)
	assert np.allclose(traj.get_var_names(["voltage"]), data)
	row = np.array([[10+0j, 10+0j, 10+0j]], dtype=np.complex64)
	traj.set_var_names(["voltage"], row, idx=[0])
	assert np.allclose(traj.get_var_names(["voltage"], idx=[0]), row)
	i_data = np.array([[7+0j, 8+0j, 9+0j]], dtype=np.complex64)
	traj.set_var_names(["current"], i_data)
	combined = traj.get_var_names(["voltage", "current"])
	assert combined.shape == (3, 3)
	assert np.allclose(combined[2:3, :], i_data)
	print("test_get_set_var_names PASSED")

def test_const_load_projection_refactored():
	S = 2.1 + 0.5j
	load = ConstPowerLoad(lambda t: S)
	traj = Trajectory(0.03, 0.01, {"voltage": 1, "current": 1, "power": 1})
	traj.set_constant(["voltage"], [1.5 + 1j])
	traj.set_constant(["current"], [2.2 - 0.3j])
	traj.set_constant(["power"], [0.0 + 0j])
	projected = load.project(traj)
	computed_power = projected.w['voltage'] * projected.w['current'].conjugate()
	for k in range(traj.N):
		assert abs(computed_power[0, k] - S) < 1e-4, \
			f"Power constraint violated at t={k}: {computed_power[0,k]} != {S}"
	print("test_const_load_projection_refactored PASSED")

def run_admm_feasibility_test():
	omega_s = 2*np.pi*60
	H1, D1, RD1, X1_p, Tsv1 = 8, 0, 0.04, 0.0608, 2
	H2, D2, RD2, X2_p, Tsv2 = 3.01, 0, 0.04, 0.1813, 2

	R12, R23, R13 = 0.01, 0.02, 0.01
	X12, X23, X13 = 0.085, 0.161, 0.092
	ysh12, ysh23, ysh13 = 0.088j, 0.153j, 0.079j

	V10, theta10 = 1.04, 0.0
	V20, theta20 = 1.025, -0.148 * np.pi / 180
	V30, theta30 = 0.994, -7.65 * np.pi / 180

	V10C = V10 * np.exp(1j * theta10)
	V20C = V20 * np.exp(1j * theta20)


	S10 = 1.597 + 0.452j
	S20 = 0.791 - 0.279j
	S30 = -2.35 - 0.5j
	PC1 = np.real(S10)
	PC2 = np.real(S20)

	I10 = np.conj(S10 / (V10 * np.exp(1j * theta10)))
	I20 = np.conj(S20 / (V20 * np.exp(1j * theta20)))
	I30 = np.conj(S30 / (V30 * np.exp(1j * theta30)))

	E1 = V10 * np.exp(1j * theta10) + 1j*X1_p * I10
	E2 = V20 * np.exp(1j * theta20) + 1j*X2_p * I20
	E10, delta10 = np.abs(E1), np.angle(E1)
	E20, delta20 = np.abs(E2), np.angle(E2)

	Ybus = np.array([
		[1/(R12+1j*X12) + 1/(R13+1j*X13) + ysh12+ysh13, -1/(R12+1j*X12), -1/(R13+1j*X13)],
		[-1/(R12+1j*X12), 1/(R12+1j*X12) + 1/(R23+1j*X23) + ysh12+ysh23, -1/(R23+1j*X23)],
		[-1/(R13+1j*X13), -1/(R23+1j*X23), 1/(R13+1j*X13) + 1/(R23+1j*X23) + ysh13+ysh23]
	], dtype=complex)

	P_init, Q_init = np.real(S30), np.imag(S30)
	P_post = -2.45

	g1 = Generator(E10, 1j*X1_p, H1, D1, Tsv1, RD1, delta10, PC1, V10C, I10, S10)
	g2 = Generator(E20, 1j*X2_p, H2, D2, Tsv2, RD2, delta20, PC2, V20C, I20, S20)
	l1 = ConstPowerLoad(lambda t: P_init + 1j*Q_init if t < 0.5 else P_post + 1j*Q_init)
	Bi = BusBehaviours([g1, g2], [l1])

	gen_costs = [1.2, 1.0]
	P_min = np.array([0.2, 0.2])
	P_max = np.array([2.0, 2.0])
	V_max = np.array([1.2, 1.2, 1.9])

	initial_traj = Trajectory(2.5, 0.01, {
		"voltage": 3, "current": 3, "power": 3,
		"delta": 2, "omega": 2, "Tm": 2
	})
	initial_traj.set_constant(["voltage"], [V10*np.exp(1j*theta10), V20*np.exp(1j*theta20), V30*np.exp(1j*theta30)])
	initial_traj.set_constant(["current"], [I10, I20, I30])
	initial_traj.set_constant(["delta"], [delta10, delta20])
	initial_traj.set_constant(["omega"], [omega_s, omega_s])
	initial_traj.set_constant(["Tm"], [PC1, PC2])
	initial_traj.set_constant(["power"], [S10, S20, S30])

	obj = Objective(Ybus, gen_costs, P_min, P_max, V_max, initial_traj)

	primal_residuals = []
	def cb(iteration, x, z, _u):
		primal_residuals.append((x - z).norm())
		print(f"ADMM iteration {iteration+1}: primal residual = {primal_residuals[-1]:.4e}")

	print("Running ADMM...")
	sol = admm(obj, Bi, initial_traj, threshold=1e-3, max_iterations=15, callback=cb)

	t_vec = np.arange(sol.N) * sol.dt
	omega1 = np.real(sol.w["omega"][0, :])
	omega2 = np.real(sol.w["omega"][1, :])
	delta1 = np.real(sol.w["delta"][0, :])
	delta2 = np.real(sol.w["delta"][1, :])
	Tm1    = np.real(sol.w["Tm"][0, :])
	Tm2    = np.real(sol.w["Tm"][1, :])
	V_mag  = np.abs(sol.w["voltage"])
	P      = np.real(sol.w["power"])
	Q      = np.imag(sol.w["power"])

	kcl_res = np.linalg.norm(Ybus @ sol.w["voltage"] - sol.w["current"], axis=0)
	load_power_computed = sol.w["voltage"][2, :] * np.conj(sol.w["current"][2, :])
	load_power_ref = np.array([l1.S(k * sol.dt) for k in range(sol.N)])
	load_res = np.abs(load_power_computed - load_power_ref)

	# Figure 1: Generator dynamics
	fig1, axs = plt.subplots(2, 2, figsize=(12, 8))
	fig1.suptitle("Generator Dynamics")

	axs[0, 0].plot(t_vec, omega1, label="Gen 1")
	axs[0, 0].plot(t_vec, omega2, label="Gen 2")
	axs[0, 0].axhline(omega_s, color='k', linestyle='--', linewidth=0.8, label="ωs")
	axs[0, 0].set_ylabel("ω (rad/s)")
	axs[0, 0].set_title("Rotor Speed")
	axs[0, 0].legend()

	axs[0, 1].plot(t_vec, np.degrees(delta1), label="Gen 1")
	axs[0, 1].plot(t_vec, np.degrees(delta2), label="Gen 2")
	axs[0, 1].set_ylabel("δ (deg)")
	axs[0, 1].set_title("Rotor Angle")
	axs[0, 1].legend()

	axs[1, 0].plot(t_vec, Tm1, label="Gen 1")
	axs[1, 0].plot(t_vec, Tm2, label="Gen 2")
	axs[1, 0].set_ylabel("Tm (pu)")
	axs[1, 0].set_title("Mechanical Torque")
	axs[1, 0].legend()

	for bus in range(3):
		axs[1, 1].plot(t_vec, V_mag[bus, :], label=f"Bus {bus+1}")
	axs[1, 1].set_ylabel("|V| (pu)")
	axs[1, 1].set_title("Bus Voltage Magnitudes")
	axs[1, 1].legend()

	for ax in axs.flat:
		ax.set_xlabel("Time (s)")
		ax.axvline(0.5, color='r', linestyle=':', linewidth=0.8)
		ax.grid(True)
	plt.tight_layout()

	# Figure 2: Power
	fig2, axs2 = plt.subplots(2, 1, figsize=(10, 7))
	fig2.suptitle("Bus Power")

	for bus in range(3):
		axs2[0].plot(t_vec, P[bus, :], label=f"Bus {bus+1}")
	axs2[0].axvline(0.5, color='r', linestyle=':', linewidth=0.8, label="disturbance")
	axs2[0].set_ylabel("P (pu)")
	axs2[0].set_title("Real Power")
	axs2[0].legend()
	axs2[0].grid(True)

	for bus in range(3):
		axs2[1].plot(t_vec, Q[bus, :], label=f"Bus {bus+1}")
	axs2[1].axvline(0.5, color='r', linestyle=':', linewidth=0.8)
	axs2[1].set_ylabel("Q (pu)")
	axs2[1].set_title("Reactive Power")
	axs2[1].set_xlabel("Time (s)")
	axs2[1].legend()
	axs2[1].grid(True)
	plt.tight_layout()

	# Figure 3: ADMM convergence and feasibility
	fig3, axs3 = plt.subplots(3, 1, figsize=(10, 9))
	fig3.suptitle("ADMM Convergence and Feasibility")

	axs3[0].semilogy(primal_residuals, marker='o', markersize=3)
	axs3[0].axhline(1e-3, color='r', linestyle='--', label="threshold")
	axs3[0].set_xlabel("Iteration")
	axs3[0].set_ylabel("‖x − z‖")
	axs3[0].set_title("Primal Residual")
	axs3[0].legend()
	axs3[0].grid(True)

	axs3[1].semilogy(t_vec, kcl_res + 1e-16)
	axs3[1].set_xlabel("Time (s)")
	axs3[1].set_ylabel("‖Ybus·V − I‖")
	axs3[1].set_title("KCL Residual over Time")
	axs3[1].grid(True)

	axs3[2].semilogy(t_vec, load_res + 1e-16)
	axs3[2].set_xlabel("Time (s)")
	axs3[2].set_ylabel("|V·I* − S_load|")
	axs3[2].set_title("Load Power Constraint Residual")
	axs3[2].grid(True)

	plt.tight_layout()
	plt.show()


# test_get_set_var_names()
# test_const_load_projection_refactored()
run_admm_feasibility_test()