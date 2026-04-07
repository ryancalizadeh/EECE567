import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Callable
import casadi as ca


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

class Network(Projectable):
	"""
	A class representing a network with a given number of nodes and Ybus matrix.

	Parameters
	----------
	n : int
		Number of nodes in the network.
	Ybus : numpy.ndarray
		The admittance matrix of the network.
	"""

	def __init__(self, n, Ybus):
		self.n = n
		self.Ybus = Ybus

		self.gram_matrix_inv = np.linalg.inv(Ybus @ Ybus.conj().T + np.eye(n))

		A = np.eye(n) - Ybus.conj().T @ self.gram_matrix_inv @ Ybus
		B = Ybus.conj().T @ self.gram_matrix_inv
		C = self.gram_matrix_inv @ Ybus
		D = np.eye(n) - self.gram_matrix_inv
		self.projector = np.block([[A, B], [C, D]])

	def project(self, trajectory: Trajectory) -> Trajectory:
		"""
		Projects the given trajectory onto the network

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
		VI = ret.get_var_names(["voltage", "current"])
		projected_VI = self.projector @ VI
		ret.set_var_names(["voltage", "current"], projected_VI)

		return ret

class Generator(Projectable):
	"""
	A class implementing projections onto a single classical generators behaviour
	"""

	def __init__(self, E, Zd, H, D, Tsv, R, delta0, Pc, max_iter=20, tol=1e-5):
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

		# Total number of variables: delta, omega, Tm, V_re, V_im, I_re, I_im
		n_vars = 7

		# Extract measurement data
		V_traj = trajectory.get_var_names(["voltage"])
		I_traj = trajectory.get_var_names(["current"])
		V_traj_re = np.real(V_traj)
		V_traj_im = np.imag(V_traj)
		I_traj_re = np.real(I_traj)
		I_traj_im = np.imag(I_traj)

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

		# Build constraints vector
		constraints_list = []
		
		# Initial conditions
		constraints_list.append(delta[0])
		constraints_list.append(omega[0] - self.ws)
		constraints_list.append(Tm[0] - self.Pc)
		
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
		
		constraints_vec = ca.vertcat(*constraints_list)

		lbg = np.zeros(constraints_vec.shape[0])
		ubg = np.zeros(constraints_vec.shape[0])

		# Objective function: minimize ||w - trajectory||_2^2
		obj = ca.sumsqr(V_re - V_traj_re) + ca.sumsqr(V_im - V_traj_im) + ca.sumsqr(I_re - I_traj_re) + ca.sumsqr(I_im - I_traj_im)

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
		
		x_init = np.column_stack([delta_guess, omega_guess, Tm_guess, V_re_guess, V_im_guess, I_re_guess, I_im_guess])
		initial_guess = x_init.flatten()

		# Solve NLP
		print("Solving NLP...")
		sol = nlp_solver(x0=initial_guess, lbg=lbg, ubg=ubg)

		# Extract solution and reshape back to 2D form
		sol_x_flat = sol['x'].full().flatten()
		sol_x = sol_x_flat.reshape((n_vars, N))
		
		sol_delta = sol_x[0, :]
		sol_omega = sol_x[1, :]
		sol_Tm = sol_x[2, :]
		sol_V_re = sol_x[3, :]
		sol_V_im = sol_x[4, :]
		sol_I_re = sol_x[5, :]
		sol_I_im = sol_x[6, :]
		sol_V = sol_V_re + 1j*sol_V_im
		sol_I = sol_I_re + 1j*sol_I_im

		# Plot omega
		plt.figure(figsize=(10, 6))
		plt.plot(np.arange(N)*dt, sol_omega, label='Projected Omega')
		plt.axhline(self.ws, color='r', linestyle='--', label='Synchronous Speed')
		plt.xlabel('Time (s)')
		plt.ylabel('Rotor Speed (rad/s)')
		plt.title('Projected Rotor Speed over Time')
		plt.legend()
		plt.grid(True)
		plt.show()


		ret.set_var_names(["voltage"], sol_V)
		ret.set_var_names(["current"], sol_I)

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
			V0 = ret.get_restriction(["voltage"], n=[n])
			I0 = ret.get_restriction(["current"], n=[n])

			s = self.S(n * ret.dt)

			# 1. Handle the degenerate case s = 0
			if abs(s) < 1e-15:
				print("Warning: Load power is zero, projecting to zero current.")
				ret.set_restriction(["current"], np.zeros_like(I0), n=[n])

			# 2. Initial Guess
			# Starting at V0 is generally robust for this manifold
			v = V0
			
			# Precompute constants
			s_mag_sq = abs(s)**2
			s_conj_i0_conj = s.conjugate() * I0.conjugate()

			delta = np.zeros_like(v)

			for i in range(self.max_iter):
				print(f"Iteration {i+1}/{self.max_iter}")

				v_conj = v.conjugate()

				h = (v**2 * v_conj**2) - (V0 * v * v_conj**2) - s_mag_sq + (s_conj_i0_conj * v)
				h_conj = h.conjugate()

				# dh/dv
				dh_dv = 2*v*(v_conj**2) - V0*(v_conj**2) + s_conj_i0_conj
				# dh/dv_conj
				dh_dvc = 2*(v**2)*v_conj - 2*V0*v*v_conj
				
				# The Jacobian matrix for the system [h, h_conj]^T
				# J = [[dh/dv, dh/dv_conj], [dh_conj/dv, dh_conj/dv_conj]]
				# Note: dh_conj/dv_conj is (dh/dv).conj() and dh_conj/dv is (dh/dv_conj).conj()
				J = np.array([
					[dh_dv, dh_dvc],
					[dh_dvc.conjugate(), dh_dv.conjugate()]
				])

				# Step calculation: delta = -J^-1 * [h, h_conj]
				try:
					delta = np.linalg.solve(J, -np.array([h, h_conj]))
				except np.linalg.LinAlgError:
					# Handle singular Jacobian if v hits a critical point
					# Print with high precision to understand how close we are to the singularity
					print(f"Warning: Jacobian is singular, stopping iteration with error delta = {delta[0, 0]:.8e}")
					break
					
				v += delta[0]

				# Convergence check
				print("Delta: ", abs(delta[0]))
				if abs(delta[0]) < self.tol:
					break

			# Calculate final I from the constraint VI* = s => I = (s/V)*
			final_v = v
			final_i = (s / final_v).conjugate()
			
			ret.set_restriction(["voltage"], final_v, n=[n])
			ret.set_restriction(["current"], final_i, n=[n])

		return ret

class BusBehaviours(Projectable):
	def __init__(self, gens: list[Generator], loads: list[ConstPowerLoad]):
		self.gens = gens
		self.loads = loads

	def project(self, trajectory: Trajectory) -> Trajectory:
		ret = trajectory.copy()
		for i, gen in enumerate(self.gens):
			print(f"Projecting onto generator {i+1}/{len(self.gens)}")
			t = trajectory.get_subtrajectory(["voltage", "current"], idx=[i])
			projected_t = gen.project(t)
			ret.set_subtrajectory(["voltage", "current"], projected_t, idx=[i])
		for i, load in enumerate(self.loads, start=len(self.gens)):
			print(f"Projecting onto load {i+1}/{len(self.loads)}")
			t = trajectory.get_subtrajectory(["voltage", "current"], idx=[i])
			projected_t = load.project(t)
			ret.set_subtrajectory(["voltage", "current"], projected_t, idx=[i])
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

	def get_subtrajectory(self, var_names: list[str], idx=None) -> Trajectory:
		"""
		Returns a new Trajectory object containing only the specified variables and indices.
		
		Parameters
		----------
		var_name : list[str]
			The names of the variables to extract.
		idx : list[int], optional
			The indices of the rows to extract. If None, extracts all rows.
		"""

		sub_vars = {var_name: self.vars[var_name] if idx is None else len(idx) for var_name in var_names}
		sub_traj = Trajectory(self.T, self.dt, sub_vars)
		for var_name in var_names:
			if idx is None:
				sub_traj.w[var_name] = self.w[var_name]
			else:
				sub_traj.w[var_name] = self.w[var_name][idx, :]
		return sub_traj
	
	def set_subtrajectory(self, var_names: list[str], sub_traj: Trajectory, idx=None) -> None:
		"""
		Sets the specified variables and indices in the current trajectory to the values from the given sub-trajectory.
		
		Parameters
		----------
		var_name : list[str]
			The names of the variables to set.
		sub_traj : Trajectory
			The trajectory containing the values to set.
		idx : list[int], optional
			The indices of the rows to set. If None, sets all rows.
		"""

		for var_name in var_names:
			if idx is None:
				self.w[var_name] = sub_traj.w[var_name]
			else:
				self.w[var_name][idx, :] = sub_traj.w[var_name]

	def get_restriction(self, var_names: list[str], n: list[int], idx=None) -> np.ndarray:
		"""
		Returns the values of the specified variables at the given time-step indices and indices.
		
		Parameters
		----------
		var_name : list[str]
			The names of the variables to extract.
		n : list[int]
			The time-step indices at which to extract the values.
		idx : list[int], optional
			The indices of the rows to extract. If None, extracts all rows.
		"""

		time_indices = n
		sub_trajectories = []
		for var_name in var_names:
			if idx is None:
				sub_trajectories.append(self.w[var_name][:, time_indices])
			else:
				sub_trajectories.append(self.w[var_name][idx, :][:, time_indices])
		return np.vstack(sub_trajectories)

	def set_restriction(self, var_names: list[str], values: np.ndarray, n: list[int], idx=None) -> None:
		"""
		Sets the values of the specified variables at the given time-step indices and indices.

		Parameters
		----------
		var_name : list[str]
			The names of the variables to set.
		values : np.ndarray
			The values to set for the specified variables.
		n : list[int]
			The time-step indices at which to set the values.
		idx : list[int], optional
			The indices of the rows to set. If None, sets all rows.
		"""
		time_indices = n
		start_row = 0
		for var_name in var_names:
			size = self.vars[var_name] if idx is None else len(idx)
			if idx is None:
				self.w[var_name][:, time_indices] = values[start_row:start_row+size, :]
			else:
				self.w[var_name][idx, :][:, time_indices] = values[start_row:start_row+size, :]
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

def dykstra(sets: list[Projectable], x0: Trajectory, threshold=1e-6, max_iterations=1000) -> Trajectory:
	increments = []
	for _ in sets:
		increments.append(Trajectory(x0.T, x0.dt, x0.vars))

	current_x = x0
	
	for iteration in range(max_iterations):
		x_start_of_cycle = current_x

		for i, p_set in enumerate(sets):
			print(f"Projecting onto set {i+1}/{len(sets)}")

			# 1. Add the increment to the current point
			w = current_x + increments[i]

			# 2. Project onto the current set
			projected_x = p_set.project(w)

			# 3. Update the increment for this specific set
			increments[i] = w - projected_x

			# 4. Update the current state
			current_x = projected_x

		# 5. Check for convergence
		diff = current_x - x_start_of_cycle

		print(f"Iteration {iteration + 1}: Norm of difference = {diff.norm()}")
		
		if diff.norm() < threshold:
			break

	return current_x

# Testing
def test_dykstra():
	class Circle(Projectable):
		def __init__(self, radius):
			self.radius = radius

		def project(self, trajectory: Trajectory) -> Trajectory:
			# Simple projection onto a circle of given radius
			norm = trajectory.norm()
			if norm > self.radius:
				projected_traj = Trajectory(trajectory.T, trajectory.dt, trajectory.vars)
				for var_name in trajectory.vars:
					projected_traj.w[var_name] = (trajectory.w[var_name] / norm) * self.radius
				return projected_traj
			else:
				return trajectory.copy()

	class Box(Projectable):
		def __init__(self, lower_bound, upper_bound):
			self.lower_bound = lower_bound
			self.upper_bound = upper_bound

		def project(self, trajectory: Trajectory) -> Trajectory:
			# Simple projection onto a box defined by lower and upper bounds
			projected_traj = Trajectory(trajectory.T, trajectory.dt, trajectory.vars)
			for var_name in trajectory.vars:
				projected_traj.w[var_name] = np.clip(trajectory.w[var_name], self.lower_bound, self.upper_bound)
			return projected_traj

	a = -0.5
	b = 1.0
	b1 = Box(a, b)
	c1 = Circle(1)

	sets = [b1, c1]
	x0 = Trajectory(1.0, 1.0, {"var1": 1, "var2": 1})
	x0.w['var1'] = np.array([[-1.0]], dtype=np.complex64)
	x0.w['var2'] = np.array([[1.0]], dtype=np.complex64)

	result = dykstra(sets, x0)

	import matplotlib.patches as patches

	# Visualization
	# Plot the trajectory and the sets for visualization
	fig, axs = plt.subplots(1, 1, figsize=(8, 8))
	axs.set_title("Dykstra's Algorithm Projection")
	axs.set_xlim(-2, 2)
	axs.set_ylim(-2, 2)
	# Plot the box
	box = patches.Rectangle((a, a), b - a, b - a, fill=False,
						edgecolor='blue', label='Box')
	axs.add_patch(box)
	# Plot the circle
	circle = patches.Circle((0, 0), 1, fill=False,
						edgecolor='red', label='Circle')
	axs.add_patch(circle)
	# Plot the trajectory
	axs.plot(result.w['var1'][0, 0].real, result.w['var2'][0, 0].real, 'go', label='Projected Trajectory')
	axs.plot(x0.w['var1'][0, 0].real, x0.w['var2'][0, 0].real, 'ro', label='Initial Trajectory')
	axs.legend()
	plt.grid()
	plt.show()

def test_network():
	n = 3
	Ybus = np.array([[6, -1, -5],
				  [-1, 7, -6],
				  [-5, -6, 11]], dtype=np.complex64)
	network = Network(n, Ybus)

	vars = {
		"voltage": n,
		"current": n
	}
	traj = Trajectory(0.1, 0.05, vars)
	traj.w['voltage'] = np.array([[1.0 + 1.0j, 1.02 + 0.8j], [1.0 + 0.5j, 1.2 + 0.6j], [1.0 + 0.3j, 1.2 + 0.4j]], dtype=np.complex64)
	traj.w['current'] = np.array([[0.0 + 0.1j, 0.5 + 0.2j], [0.0 + 0.15j, 0.4 + 0.25j], [0.0 + 0.2j, 0.3 + 0.1j]], dtype=np.complex64)

	projected_traj = network.project(traj)

	print("Projected Voltages:")
	print(projected_traj.w['voltage'])
	print("Projected Currents:")
	print(projected_traj.w['current'])

	print("Residual:")
	print(np.linalg.norm(Ybus @ projected_traj.w['voltage'] - projected_traj.w['current']))

def test_dynamics():
	omega_s = 2*np.pi*60
	H1 = 8
	D1 = 0
	RD1 = 0.04
	X1_p = 0.0608
	H2 = 3.01
	D2 = 0
	RD2 = 0.04
	X2_p = 0.1813

	Tsv1 = 2
	Tsv2 = 2

	R12 = 0.01
	R23 = 0.02
	R13 = 0.01
	X12 = 0.085
	X23 = 0.161
	X13 = 0.092
	ysh12 = 0.088j
	ysh23 = 0.153j
	ysh13 = 0.079j

	V10 = 1.04
	theta10 = 0.0
	V20 = 1.025
	theta20 = -0.148 * np.pi / 180
	V30 = 0.994
	theta30 = -7.65 * np.pi / 180

	V10_complex = V10 * np.exp(1j * theta10)
	V20_complex = V20 * np.exp(1j * theta20)
	V30_complex = V30 * np.exp(1j * theta30)

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
	E10 = np.abs(E1)
	E20 = np.abs(E2)

	delta10 = np.angle(E1)
	delta20 = np.angle(E2)

	omega10 = omega_s
	omega20 = omega_s
	Ybus = np.array([
	[1/(R12 + 1j*X12) + 1/(R13 + 1j*X13) + ysh12 + ysh13, -1/(R12 + 1j*X12), -1/(R13 + 1j*X13)],
	[-1/(R12 + 1j*X12), 1/(R12 + 1j*X12) + 1/(R23 + 1j*X23) + ysh12 + ysh23, -1/(R23 + 1j*X23)],
	[-1/(R13 + 1j*X13), -1/(R23 + 1j*X23), 1/(R13 + 1j*X13) + 1/(R23 + 1j*X23) + ysh13 + ysh23]
	], dtype=complex)

	P_init = np.real(S30)
	Q_init = np.imag(S30)
	P_post = -2.45
	Q_post = Q_init

	print(E10, X1_p, H1, D1, Tsv1, RD1, delta10, PC1)

	g1 = Generator(E10, 1j*X1_p, H1, D1, Tsv1, RD1, delta10, PC1)
	g2 = Generator(E20, 1j*X2_p, H2, D2, Tsv2, RD2, delta20, PC2)
	l1 = ConstPowerLoad(lambda t: P_init + 1j*Q_init if t < 0.5 else P_post + 1j*Q_post)

	Bi = BusBehaviours([g1, g2], [l1])
	Bnet = Network(3, Ybus)

	initial_traj = Trajectory(5.0, 0.01, {"voltage": 3, "current": 3})
	# Initialize with constant values for each 
	initial_traj.set_constant(["voltage"], [V10_complex, V20_complex, V30_complex])
	initial_traj.set_constant(["current"], [I10, I20, I30])

	print("Initial Trajectory:")
	print("Voltages:")
	print(initial_traj.w['voltage'])
	print("Currents:")
	print(initial_traj.w['current'])


	# Test generator dynamics and aes

	print("f(x)")
	print(g1.f(np.array([0, omega10, PC1, V10_complex.real, V10_complex.imag, I10.real, I10.imag])))
	print("ae(x)")
	print(g1.ae(np.array([0, omega10, PC1, V10_complex.real, V10_complex.imag, I10.real, I10.imag])))

	projected_traj = dykstra([Bi, Bnet], initial_traj)

	# Plotting results
	time_points = np.arange(0, projected_traj.T, projected_traj.dt)
	plt.figure(figsize=(12, 8))
	plt.subplot(2, 1, 1)
	plt.plot(time_points, projected_traj.w['voltage'][0, :].real, label='Voltage at Bus 1')
	plt.plot(time_points, projected_traj.w['voltage'][1, :].real, label='Voltage at Bus 2')
	plt.plot(time_points, projected_traj.w['voltage'][2, :].real, label='Voltage at Bus 3')
	plt.title('Voltage Trajectories')
	plt.xlabel('Time (s)')
	plt.ylabel('Voltage Magnitude')
	plt.legend()
	plt.grid()

	plt.subplot(2, 1, 2)
	plt.plot(time_points, projected_traj.w['current'][0, :].real, label='Current at Bus 1')
	plt.plot(time_points, projected_traj.w['current'][1, :].real, label='Current at Bus 2')
	plt.plot(time_points, projected_traj.w['current'][2, :].real, label='Current at Bus 3')
	plt.title('Current Trajectories')
	plt.xlabel('Time (s)')
	plt.ylabel('Current Magnitude')
	plt.legend()
	plt.grid()

	plt.tight_layout()
	plt.show()

def test_const_load_projection():
	load = ConstPowerLoad(lambda t: 2.1 + 0.5j)
	traj = Trajectory(0.03, 0.01, {"voltage": 1, "current": 1})
	traj.set_constant(["voltage"], [1.5 + 1j])
	traj.set_constant(["current"], [2.2 - 0.3j])
	projected_traj = load.project(traj)
	print("Projected Voltage:", projected_traj.w['voltage'])
	print("Projected Current:", projected_traj.w['current'])
	print("Power at each time step:", projected_traj.w['voltage'] * projected_traj.w['current'].conjugate())

# test_const_load_projection()
test_dynamics()
# test_dykstra()