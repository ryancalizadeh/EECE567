import numpy as np
from abc import ABC, abstractmethod
import casadi as ca
import matplotlib.pyplot as plt
from Trajectory import Trajectory
from SysParams import SysParams
from Proxable import Proxable


class Generator(Proxable):
	"""
	A class implementing projections onto a single classical generator + fast governor behaviour
	"""

	def __init__(self, E, Zd, H, D, Tsv, R, delta0, Pc, V0, I0, S0, max_iter=20, tol=1e-5, Pc_min=-np.inf, Pc_max=np.inf):
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
		self.Pc_min = Pc_min
		self.Pc_max = Pc_max

	def prox(self, trajectory: Trajectory, rho: float = 1.0) -> Trajectory:
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

		# Total number of variables: delta, omega, Tm, V_re, V_im, I_re, I_im, P, Q, Pc
		n_vars = 10

		# Extract measurement data
		V_traj = trajectory.get_var_names(["voltage"])
		I_traj = trajectory.get_var_names(["current"])
		delta_traj = trajectory.get_var_names(["delta"])
		omega_traj = trajectory.get_var_names(["omega"])
		Tm_traj = trajectory.get_var_names(["Tm"])
		S_traj = trajectory.get_var_names(["power"])
		Pc_traj = trajectory.get_var_names(["Pc"])

		V_traj_re = np.real(V_traj)
		V_traj_im = np.imag(V_traj)
		I_traj_re = np.real(I_traj)
		I_traj_im = np.imag(I_traj)
		delta_traj = np.real(delta_traj)
		omega_traj = np.real(omega_traj)
		Tm_traj = np.real(Tm_traj)
		P_traj = np.real(S_traj)
		Q_traj = np.imag(S_traj)
		Pc_traj = np.real(Pc_traj)

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
		Pc = x_matrix[9, :]

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
			f_Tm_k = (-Tm[k] + Pc[k] - 1/self.R * (omega[k]/self.ws - 1)) / self.Tsv
			f_Tm_k1 = (-Tm[k+1] + Pc[k+1] - 1/self.R * (omega[k+1]/self.ws - 1)) / self.Tsv
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
		f_delta_N = delta[k] - delta[k-1]
		# E_k_re = self.E * ca.cos(delta[k] + self.delta0)
		# E_k_im = self.E * ca.sin(delta[k] + self.delta0)
		# Pe_k = E_k_re * I_re[k] + E_k_im * I_im[k]
		# f_omega_N = (Tm[k] - Pe_k - self.D*(omega[k]/self.ws - 1)) * self.ws/(2*self.H)
		# f_Tm_N = (-Tm[k] + Pc[k] - 1/self.R * (omega[k]/self.ws - 1)) / self.Tsv
		f_omega_N = omega[k] - omega[k-1]
		f_Tm_N = Tm[k] - Tm[k-1]

		# constraints_list.append(f_delta_N)   # delta_dot = 0
		# constraints_list.append(f_omega_N)   # omega_dot = 0
		# constraints_list.append(f_Tm_N)      # Tm_dot = 0

		constraints_list.append(omega[k] - self.ws)

		
		constraints_vec = ca.vertcat(*constraints_list)

		lbg = np.zeros(constraints_vec.shape[0])
		ubg = np.zeros(constraints_vec.shape[0])

		# Objective function: minimize ||w - trajectory||_2^2
		obj = ca.sumsqr(V_re - V_traj_re) + ca.sumsqr(V_im - V_traj_im) + ca.sumsqr(I_re - I_traj_re) + ca.sumsqr(I_im - I_traj_im) + ca.sumsqr(omega - omega_traj) + ca.sumsqr(Tm - Tm_traj) + ca.sumsqr(delta - delta_traj) + ca.sumsqr(P - P_traj) + ca.sumsqr(Q - Q_traj) + ca.sumsqr(Pc - Pc_traj)


		opts = {
			'ipopt.print_level': 0, # 0 for silent, 5 for detailed output
			'print_time': 0,
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
		Pc_guess = Pc_traj.flatten()

		x_init = np.column_stack([delta_guess, omega_guess, Tm_guess, V_re_guess, V_im_guess, I_re_guess, I_im_guess, P_guess, Q_guess, Pc_guess])
		initial_guess = x_init.flatten()

		lbx = np.full((N, n_vars), -np.inf)
		ubx = np.full((N, n_vars), np.inf)
		lbx[:, 1] = self.ws - 0.08
		ubx[:, 1] = self.ws + 0.08
		lbx[:, 9] = self.Pc_min
		ubx[:, 9] = self.Pc_max

		# Solve NLP
		# print("Solving NLP...")
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
		sol_Pc = sol_x[9, :].reshape(1, -1)

		ret.set_var_names(["voltage"], sol_V)
		ret.set_var_names(["current"], sol_I)
		ret.set_var_names(["delta"], sol_delta)
		ret.set_var_names(["omega"], sol_omega)
		ret.set_var_names(["Tm"], sol_Tm)
		ret.set_var_names(["power"], sol_S)
		ret.set_var_names(["Pc"], sol_Pc)

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