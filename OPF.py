from SysParams import SysParams
import casadi as ca
import numpy as np

def solve_opf(sys_params: SysParams, post=False) -> dict:
	"""
	Solve the static AC Optimal Power Flow (OPF) problem using CasADI + IPOPT.
	
	Formulation:
	  min Σ_i (P_gen[i])^2 / (1/cost[i])
	  s.t.
	    Ybus @ V = I                                           (AC power flow / KCL)
	    Re(V[load_j] * conj(I[load_j])) = Re(S_load[j])       (fixed load real power)
	    Im(V[load_j] * conj(I[load_j])) = Im(S_load[j])       (fixed load reactive power)
	    With bounds:
	      P_min[i] <= Re(V[i] * conj(I[i])) <= P_max[i]       (generator bounds)
	      0 <= |V[i]| <= V_max[i]                             (voltage bounds)
	
	Parameters
	----------
	sys_params : SysParams
		System parameters (Ybus, constraints, costs, etc.)
	
	Returns
	-------
	dict
		Solution dictionary with keys:
		- 'V': complex voltage vector (n_buses,)
		- 'I': complex current vector (n_buses,)
		- 'P_gen': real power generation vector (n_gens,)
		- 'status': solver status string
	"""

	n_buses = sys_params.n_buses
	n_gens = sys_params.n_gens

	if post:
		S_load = sys_params.P_load_post + 1j * sys_params.Q_load0
	else:
		S_load = sys_params.S_load0

	# Create optimization variables (as 2D matrix for easier indexing)
	V = ca.SX.sym('V', n_buses) # type: ignore
	theta = ca.SX.sym('theta', n_buses) # type: ignore
	P = ca.SX.sym('P', n_gens) # type: ignore
	Q = ca.SX.sym('Q', n_gens) # type: ignore
	x = ca.vertcat(V, theta, P, Q)
	
	# Constraints list: equality constraints only
	constraints = []
	constraints.append(theta[0])

	# Power flow constraints:
	Ybus = sys_params.Ybus
	for bus in range(n_buses):
		Pi = P[bus] if bus < n_gens else S_load.real
		Qi = Q[bus] if bus < n_gens else S_load.imag
		for other_bus in range(n_buses):
			g = np.real(Ybus[bus, other_bus])
			b = np.imag(Ybus[bus, other_bus])
			Pi -= V[bus] * V[other_bus] * (g*ca.cos(theta[bus]-theta[other_bus]) + b*ca.sin(theta[bus]-theta[other_bus]))
			Qi -= V[bus] * V[other_bus] * (g*ca.sin(theta[bus]-theta[other_bus]) - b*ca.cos(theta[bus]-theta[other_bus]))
		constraints.append(Pi)
		constraints.append(Qi)

	# Combine constraints
	g = ca.vertcat(*constraints)

	cost = 0.5 * ca.dot(P**2, sys_params.gen_costs)  # Quadratic cost for generators

	# Create NLP
	nlp = {
		'x': x,
		'f': cost,
		'g': g,
	}

	# Solve with IPOPT
	opts = {
		'ipopt': {
			'max_iter': 2000,
			'print_level': 0,
			'sb': 'yes',
		},
		'print_time': 0,
	}
	solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

	# Initial guess: voltages near 1.0, angle near 0, power at load/generation targets
	x0_V = np.ones(n_buses)
	x0_theta = np.zeros(n_buses)
	x0_P = np.full(n_gens, sys_params.S_gen0.real)
	x0_Q = np.full(n_gens, sys_params.S_gen0.imag)
	x0 = np.concatenate([x0_V, x0_theta, x0_P, x0_Q])

	# Bounds: all constraints are equality (g=0)
	ubg = np.zeros(g.shape[0])
	lbg = np.zeros(g.shape[0])

	# Variable bounds: [V, theta, P, Q]
	lbx = np.concatenate([
		np.full(n_buses, 0.5),   # V >= 0.5
		np.full(n_buses, -np.pi/2),  # theta >= -pi/2
		np.full(n_gens, 0.1),    # P >= 0.1
		np.full(n_gens, -np.inf),
	])
	ubx = np.concatenate([
		np.full(n_buses, np.inf),
		np.full(n_buses, np.pi/2),   # theta <= +pi/2
		np.full(n_gens, 5.0),    # P <= 5.0
		np.full(n_gens, np.inf),
	])

	# Solve
	sol = solver(x0=x0, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)

	# Extract solution
	x_sol = sol['x'].full().flatten()
	V_sol = x_sol[0:n_buses]
	theta_sol = x_sol[n_buses:2*n_buses]
	P_sol = x_sol[2*n_buses:2*n_buses+n_gens]
	Q_sol = x_sol[2*n_buses+n_gens:]

	return {
		'V': V_sol,
		'theta': theta_sol,
		'P': P_sol,
		'Q': Q_sol,
		'status': str(solver.stats()['return_status']),
	}

def ic_from_opf(opf_sol: dict, sys_params: SysParams, S_init: np.ndarray) -> dict:
	"""
	Extract initial conditions from OPF solution for ADMM initialization.
	
	Computes generator EMF angles and speeds from the OPF voltage/current solution.
	
	Parameters
	----------
	opf_sol : dict
		OPF solution from solve_opf()
	sys_params : SysParams
		System parameters
	S_init : np.ndarray
		Initial power injections (for consistency check; derived from opf_sol in practice)
	
	Returns
	-------
	dict
		Initial condition dict with keys:
		- 'voltage': complex voltage vector (n_buses,)
		- 'current': complex current vector (n_buses,)
		- 'power': complex power vector (n_buses,)
		- 'delta': rotor angle vector (n_gens,) [radians]
		- 'omega': rotor speed vector (n_gens,) [rad/s]
		- 'Tm': mechanical torque / generation setpoint (n_gens,) [pu]
		- 'Pc': governor command (n_gens,) [pu]
	"""
	V_opf = opf_sol['V']
	theta_opf = opf_sol['theta']
	P_opf = opf_sol['P']
	Q_opf = opf_sol['Q']

	V = V_opf * np.exp(1j * theta_opf)  # Reconstruct complex voltage
	I = sys_params.Ybus @ V  # Compute current from Ybus and voltage
	S = P_opf + 1j * Q_opf  # Complex power injection

	# Compute generator EMF and extract angles
	E_complex = V[:sys_params.n_gens] + 1j * sys_params.X_p * I[:sys_params.n_gens]
	delta_opf = np.angle(E_complex)

	# Initial conditions: all at steady state
	ic = {
		'voltage': V,
		'current': I,
		'power': V * np.conj(I),
		'S': S,
		'delta': delta_opf,
		'omega': np.full(sys_params.n_gens, sys_params.omega_s),
		'Tm': P_opf,
		'Pc': P_opf,
	}

	return ic