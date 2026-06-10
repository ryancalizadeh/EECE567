import os
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
from SysParams import SysParams
from Trajectory import Trajectory
from Projectable import Projectable
from Generator import Generator
from ConstPowerLoad import ConstPowerLoad
from BusBehaviours import BusBehaviours, BusBehavioursParallel
from Objective import Objective
from Solvable import Solvable
from OPF import solve_opf, ic_from_opf
#from stopping_criteria import primal_eps, dual_eps


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
	xs: list[Trajectory] = [Trajectory(z0.T, z0.dt, z0.vars, dtype=z0.dtype)]
	us: list[Trajectory] = [Trajectory(z0.T, z0.dt, z0.vars, dtype=z0.dtype)]

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
	traj = Trajectory(0.03, 0.01, {"voltage": 1, "current": 1, "power": 1}, dtype=np.complex64)
	traj.set_constant(["voltage"], [1.5 + 1j])
	traj.set_constant(["current"], [2.2 - 0.3j])
	projected_traj = load.project(traj)
	print("Projected Voltage:", projected_traj.w['voltage'])
	print("Projected Current:", projected_traj.w['current'])
	print("Power at each time step:", projected_traj.w['voltage'] * projected_traj.w['current'].conjugate())

def test_get_set_var_names():
	traj = Trajectory(0.03, 0.01, {"voltage": 2, "current": 1}, dtype=np.complex64)
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
	traj = Trajectory(0.03, 0.01, {"voltage": 1, "current": 1, "power": 1}, dtype=np.complex64)
	traj.set_constant(["voltage"], [1.5 + 1j])
	traj.set_constant(["current"], [2.2 - 0.3j])
	traj.set_constant(["power"], [0.0 + 0j])
	projected = load.project(traj)
	computed_power = projected.w['voltage'] * projected.w['current'].conjugate()
	for k in range(traj.N):
		assert abs(computed_power[0, k] - S) < 1e-4, \
			f"Power constraint violated at t={k}: {computed_power[0,k]} != {S}"
	print("test_const_load_projection_refactored PASSED")


def admm_test(n_buses: int = 24, seq_and_parallel=True):
	assert n_buses % 2 == 0, "n_buses must be even"
	
	# Initialize system parameters
	sys_params = SysParams(n_buses)
	n_buses = sys_params.n_buses
	n_gens = sys_params.n_gens
	n_loads = sys_params.n_loads
	omega_s = sys_params.omega_s

	# Solve static OPF to get feasible initial conditions
	print("Solving static OPF for initial conditions...")
	opf_sol = solve_opf(sys_params)
	print(f"OPF solved with status: {opf_sol['status']}")

	# Extract initial conditions from OPF solution
	S_gen0 = sys_params.S_gen0
	S_load0 = sys_params.S_load0
	S_init = np.concatenate([
		np.full(n_gens, S_gen0),
		np.full(n_loads, S_load0)
	])
	ic = ic_from_opf(opf_sol, sys_params, S_init)

	# Verify IC feasibility
	V_opf = ic['voltage']
	I_opf = ic['current']
	Ybus = sys_params.Ybus
	kcl_residual = np.linalg.norm(Ybus @ V_opf - I_opf)
	print(f"IC KCL residual: {kcl_residual:.4e}")
	assert kcl_residual < 1e-3, f"OPF solution does not satisfy KCL: residual = {kcl_residual}"

	power_residual = np.linalg.norm(ic['power'][:n_gens] - ic['S'])
	print(f"Power residual: {power_residual:.4e}")
	assert power_residual < 1e-3, f"OPF solution does not match expected power injections: residual = {power_residual}"

	# Load disturbance at t=0.5s: each load shifts real power by dP in [-0.2, +0.2]
	rng = np.random.default_rng(42)
	dP_loads    = rng.uniform(-0.2, 0.2, n_loads)
	P_load_post = np.real(S_load0) + dP_loads
	Q_load0     = np.imag(S_load0)

	# Extract generator parameters for instantiation
	H = sys_params.H
	D = sys_params.D
	Tsv = sys_params.Tsv
	RD = sys_params.Rd
	X_p = sys_params.X_p
	P_min = sys_params.P_min
	P_max = sys_params.P_max
	V_max = sys_params.V_max
	gen_costs = sys_params.gen_costs

	# Compute E_mags for Generator instantiation (from OPF IC)
	E_complex = ic['voltage'][:n_gens] + 1j * X_p * ic['current'][:n_gens]
	E_mags = np.abs(E_complex)
	delta0s = ic['delta']
	gens = [
		Generator(E_mags[i], 1j*X_p, H, D, Tsv, RD, delta0s[i], ic['Tm'][i],
		          ic['voltage'][i], ic['current'][i], ic['power'][i], 
		          Pc_min=P_min[i], Pc_max=P_max[i])
		for i in range(n_gens)
	]
	loads = [
		ConstPowerLoad(
			lambda t, p0=np.real(S_load0), p1=float(P_load_post[j]), q=Q_load0:
				p0 + 1j*q if t < 0.5 else p1 + 1j*q
		)
		for j in range(n_loads)
	]

	def make_initial_traj():
		t = Trajectory(sys_params.T, sys_params.dt, {
			"voltage": n_buses, "current": n_buses, "power": n_buses,
			"delta": n_gens, "omega": n_gens, "Tm": n_gens, "Pc": n_gens,
		}, dtype=np.complex64)
		# Use OPF-derived initial conditions
		t.set_constant(["voltage"], list(ic['voltage']))
		t.set_constant(["current"], list(ic['current']))
		t.set_constant(["delta"],   list(ic['delta']))
		t.set_constant(["omega"],   list(ic['omega']))
		t.set_constant(["Tm"],      list(ic['Tm']))
		t.set_constant(["power"],   list(ic['power']))
		t.set_constant(["Pc"],      list(ic['Pc']))
		return t

	log = logging.getLogger("admm_benchmark")
	log.setLevel(logging.INFO)
	fh = logging.FileHandler("admm_benchmark.log", mode="a")
	fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
	log.addHandler(fh)

	timing_results = {}
	primal_residuals = []

	def make_cb(residuals):
		def cb(iteration, x, z, _u):
			residuals.append((x - z).norm())
			print(f"ADMM iteration {iteration+1}: primal residual = {residuals[-1]:.4e}")
		return cb

	log.info("Number of Buses: " + str(n_buses))
	test_cases = []
	if seq_and_parallel:
		test_cases.append(("BusBehaviours (sequential)", BusBehaviours(gens, loads)))
	test_cases.append(("BusBehavioursParallel (threaded)", BusBehavioursParallel(gens, loads)))
	
	for label, Bi in test_cases:
		initial_traj = make_initial_traj()
		obj = Objective(Ybus, gen_costs, P_min, P_max, V_max, initial_traj)
		residuals = []
		print(f"\nRunning ADMM with {label}...")
		t0 = time.perf_counter()
		result = admm(obj, Bi, initial_traj, threshold=1e-3, max_iterations=2000, callback=make_cb(residuals))
		elapsed = time.perf_counter() - t0
		timing_results[label] = {"time": elapsed, "iterations": len(residuals), "result": result, "residuals": residuals}
		log.info(f"{label}: {elapsed:.3f}s over {len(residuals)} iteration(s), final residual = {residuals[-1]:.4e}")
		print(f"  -> {elapsed:.3f}s")

	if seq_and_parallel:
		log.info(
			f"Speedup (sequential / parallel): "
			f"{timing_results['BusBehaviours (sequential)']['time'] / timing_results['BusBehavioursParallel (threaded)']['time']:.2f}x"
		)
	print(f"\nBenchmark results logged to admm_benchmark.log")

	# Select solution from sequential test if available, otherwise from parallel test
	if seq_and_parallel and "BusBehaviours (sequential)" in timing_results:
		sol = timing_results["BusBehaviours (sequential)"]["result"]
		primal_residuals = timing_results["BusBehaviours (sequential)"]["residuals"]
	else:
		sol = timing_results["BusBehavioursParallel (threaded)"]["result"]
		primal_residuals = timing_results["BusBehavioursParallel (threaded)"]["residuals"]

	t_vec  = np.arange(sol.N) * sol.dt
	V_mag  = np.abs(sol.w["voltage"])
	P      = np.real(sol.w["power"])
	Q      = np.imag(sol.w["power"])

	kcl_res = np.linalg.norm(Ybus @ sol.w["voltage"] - sol.w["current"], axis=0)
	load_res_per_bus = [
		np.abs(
			sol.w["voltage"][n_gens + j, :] * np.conj(sol.w["current"][n_gens + j, :])
			- np.array([loads[j].S(k * sol.dt) for k in range(sol.N)])
		)
		for j in range(n_loads)
	]
	load_res = np.max(load_res_per_bus, axis=0)

	# Figure 1: Generator dynamics
	fig1, axs = plt.subplots(2, 2, figsize=(14, 10))
	fig1.suptitle("Generator Dynamics (12-bus, 6 gens)")

	for i in range(n_gens):
		axs[0, 0].plot(t_vec, np.real(sol.w["omega"][i, :]), label=f"Gen {i+1}")
	axs[0, 0].axhline(omega_s, color='k', linestyle='--', linewidth=0.8, label="ωs")
	axs[0, 0].set_ylabel("ω (rad/s)")
	axs[0, 0].set_title("Rotor Speed")
	axs[0, 0].legend(fontsize=7)
	axs[0, 0].set_ylim(omega_s - 0.1, omega_s + 0.1)

	for i in range(n_gens):
		axs[0, 1].plot(t_vec, np.degrees(np.real(sol.w["delta"][i, :])), label=f"Gen {i+1}")
	axs[0, 1].set_ylabel("δ (deg)")
	axs[0, 1].set_title("Rotor Angle")
	axs[0, 1].legend(fontsize=7)

	for i in range(n_gens):
		axs[1, 0].plot(t_vec, np.real(sol.w["Tm"][i, :]), label=f"Gen {i+1}")
	axs[1, 0].set_ylabel("Tm (pu)")
	axs[1, 0].set_title("Mechanical Torque")
	axs[1, 0].legend(fontsize=7)

	for bus in range(n_buses):
		lbl = f"Gen {bus+1}" if bus < n_gens else f"Load {bus-n_gens+1}"
		axs[1, 1].plot(t_vec, V_mag[bus, :], label=lbl)
	axs[1, 1].set_ylabel("|V| (pu)")
	axs[1, 1].set_title("Bus Voltage Magnitudes")
	axs[1, 1].legend(fontsize=6)

	for ax in axs.flat:
		ax.set_xlabel("Time (s)")
		ax.axvline(0.5, color='r', linestyle=':', linewidth=0.8)
		ax.grid(True)
	plt.tight_layout()

	# Figure 2: Power
	fig2, axs2 = plt.subplots(2, 1, figsize=(12, 8))
	fig2.suptitle("Bus Power (12-bus)")

	for bus in range(n_buses):
		lbl = f"Gen {bus+1}" if bus < n_gens else f"Load {bus-n_gens+1}"
		axs2[0].plot(t_vec, P[bus, :], label=lbl)
	axs2[0].axvline(0.5, color='r', linestyle=':', linewidth=0.8, label="disturbance")
	axs2[0].set_ylabel("P (pu)")
	axs2[0].set_title("Real Power")
	axs2[0].legend(fontsize=6)
	axs2[0].grid(True)

	for bus in range(n_buses):
		lbl = f"Gen {bus+1}" if bus < n_gens else f"Load {bus-n_gens+1}"
		axs2[1].plot(t_vec, Q[bus, :], label=lbl)
	axs2[1].axvline(0.5, color='r', linestyle=':', linewidth=0.8)
	axs2[1].set_ylabel("Q (pu)")
	axs2[1].set_title("Reactive Power")
	axs2[1].set_xlabel("Time (s)")
	axs2[1].legend(fontsize=6)
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
	axs3[2].set_ylabel("|V·I* − S_load| (max over loads)")
	axs3[2].set_title("Load Power Constraint Residual")
	axs3[2].grid(True)

	plt.tight_layout()
	plt.show()

	return timing_results




# test_get_set_var_names()
# test_const_load_projection_refactored()
# run_admm_feasibility_test()

if __name__ == "__main__":
	import pickle
	
	times = {}
	if os.path.exists("admm_times.pkl"):
		with open("admm_times.pkl", "rb") as f:
			times = pickle.load(f)

	nbuses = 4

	timing_results = admm_test(n_buses=nbuses, seq_and_parallel=False)
	times[nbuses] = timing_results

	# Save times dict to file
	with open("admm_times.pkl", "wb") as f:
		pickle.dump(times, f)
			
