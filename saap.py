import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
from typing import cast, List, Callable
from SysParams import SysParams
from Trajectory import Trajectory
from Generator import Generator
from ConstPowerLoad import ConstPowerLoad
from BusBehaviours import BusBehavioursSerial, BusBehavioursParallel, BusBehaviours
from Objective import Objective
from Proxable import Proxable
from OPF import solve_opf, ic_from_opf
#from stopping_criteria import primal_eps, dual_eps

def rho_heuristic(iteration, rho_prev, r, s, tau=1.1, mu=100):
	if rho_prev == 0:
		return 2.0
	elif np.linalg.norm(r) > mu*np.linalg.norm(s):
		print(f"Warning: Norm of r at iteration {iteration} is {np.linalg.norm(r)}, multiplying rho by {tau} \n\n\n")
		return tau * rho_prev
	elif np.linalg.norm(s) > mu*np.linalg.norm(r):
		print(f"Warning: Norm of s at iteration {iteration} is {np.linalg.norm(s)}, dividing rho by {tau} \n\n\n")
		return rho_prev / tau
	else:
		return rho_prev
	
def base_callback(iteration, x, z, u, r, s):
	if iteration % 10 == 0:
		print(f"ADMM iteration {iteration+1}: primal residual = {r:.4e}, dual residual = {s:.4e}")

def compute_cost(x: Trajectory, sys_params: SysParams):
	"""
	Compute the cost of a solution to the admm problem
	"""

	cost = 0
	for i in range(sys_params.n_gens):
		# P_gen = Re(V_i * conj(I_i))
		P_gen = np.real(x.w["voltage"][i, :] * np.conj(x.w["current"][i, :]))
		cost += np.sum(sys_params.gen_costs[i] * (P_gen**2))
	return cost



def admm(f: Proxable, g: Proxable, z0: Trajectory, rho=lambda i, prev, r, s: 2.0, threshold=1e-3, max_iterations=100, callback=None, weights: dict | None = None):
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
	weights : dict, optional
		Per-variable diagonal preconditioning weights. If provided, the primal/dual
		residual norms used for convergence and the rho heuristic are computed with
		these weights, to stay consistent with the weighted penalty applied in f/g's prox.
	"""

	# Initialize x0, z0, mu0
	zs: list[Trajectory] = [z0.copy()]
	xs: list[Trajectory] = [Trajectory(z0.T, z0.dt, z0.vars, dtype=z0.dtype)]
	us: list[Trajectory] = [Trajectory(z0.T, z0.dt, z0.vars, dtype=z0.dtype)]

	def res_norm(traj: Trajectory):
		return traj.weighted_norm(weights) if weights is not None else traj.norm()

	rs = [res_norm(xs[-1] - zs[-1])]
	ss = [res_norm(xs[-1] - zs[-1])]
	rhos = [2.0]

	for iteration in range(max_iterations-1):
		new_rho = rho(iteration, rhos[-1], rs[-1], ss[-1])
		rhos.append(new_rho)
		# Rescale u when rho changes to keep λ = rho*u continuous
		if new_rho != rhos[-2]:
			us[-1] = us[-1] * (rhos[-2] / new_rho)
		# zs[-1].plot("Zs before proxing")
		xs.append(f.prox(zs[-1] - us[-1], rhos[-1]))
		# xs[-1].plot("Xs after proxing")
		zs.append(g.prox(xs[-1] + us[-1], rhos[-1]))
		# zs[-1].plot("Zs after proxing")
		us.append(us[-1] + (xs[-1] - zs[-1]))

		# (xs[-1] - zs[-1]).plot("Difference between Xs and Zs")

		rs.append(res_norm(xs[-1] - zs[-1]))
		ss.append(rhos[-1] * res_norm(zs[-1] - zs[-2]))

		if callback is not None:
			callback(iteration, xs[-1], zs[-1], us[-1], rs[-1], ss[-1])

		if rs[-1] < threshold and ss[-1] < threshold:
			break

	return xs[-1]

def test_const_load_projection():
	load = ConstPowerLoad(lambda t: 2.1 + 0.5j)
	traj = Trajectory(0.03, 0.01, {"voltage": 1, "current": 1, "power": 1}, dtype=np.complex64)
	traj.set_constant(["voltage"], [1.5 + 1j])
	traj.set_constant(["current"], [2.2 - 0.3j])
	projected_traj = load.prox(traj)
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
	projected = load.prox(traj)
	computed_power = projected.w['voltage'] * projected.w['current'].conjugate()
	for k in range(traj.N):
		assert abs(computed_power[0, k] - S) < 1e-4, \
			f"Power constraint violated at t={k}: {computed_power[0,k]} != {S}"
	print("test_const_load_projection_refactored PASSED")


def _setup_admm_problem(n_buses: int = 24, weights: dict | None = None):
	"""Solve OPF, build generators/loads, and return everything needed to run ADMM.

	weights : dict, optional
		Per-variable diagonal preconditioning weights to bake into the constructed
		Generators. If None, falls back to sys_params.precondition_weights().
	"""
	assert n_buses % 2 == 0, "n_buses must be even"

	sys_params = SysParams(n_buses)
	n_buses = sys_params.n_buses
	n_gens = sys_params.n_gens
	n_loads = sys_params.n_loads

	print("Solving static OPF for initial conditions...")
	opf_sol = solve_opf(sys_params)
	print(f"OPF solved with status: {opf_sol['status']}")

	print("Solving static OPF for post-increase initial conditions...")
	opf_sol_post = solve_opf(sys_params, post=True)
	print(f"OPF solved with status: {opf_sol_post['status']}")

	S_init = np.concatenate([
		np.full(n_gens, sys_params.S_gen0),
		np.full(n_loads, sys_params.S_load0),
	])
	ic = ic_from_opf(opf_sol, sys_params, S_init)
	ic_post = ic_from_opf(opf_sol_post, sys_params, S_init)

	Ybus = sys_params.Ybus
	kcl_residual = np.linalg.norm(Ybus @ ic['voltage'] - ic['current'])
	print(f"IC KCL residual: {kcl_residual:.4e}")
	assert kcl_residual < 1e-3, f"OPF solution does not satisfy KCL: residual = {kcl_residual}"

	power_residual = np.linalg.norm(ic['power'][:n_gens] - ic['S'])
	print(f"Power residual: {power_residual:.4e}")
	assert power_residual < 1e-3, f"OPF solution does not match expected power injections: residual = {power_residual}"

	X_p = sys_params.X_p
	P_min = sys_params.P_min
	P_max = sys_params.P_max
	if weights is None:
		weights = sys_params.precondition_weights()
	E_mags = np.abs(ic['voltage'][:n_gens] + 1j * X_p * ic['current'][:n_gens])
	gens = [
		Generator(E_mags[i], 1j*X_p, sys_params.H, sys_params.D, sys_params.Tsv, sys_params.Rd,
		          ic['delta'][i], ic['Tm'][i], ic['voltage'][i], ic['current'][i], ic['power'][i],
		          Pc_min=P_min[i], Pc_max=P_max[i], weights=weights)
		for i in range(n_gens)
	]
	loads = [ConstPowerLoad(sys_params.get_load_power(j)) for j in range(n_loads)]

	def make_initial_traj(warm_start_file=None, **kwargs):
		if warm_start_file is not None and os.path.exists(warm_start_file):
			with open(warm_start_file, "rb") as f:
				sols = pickle.load(f)
				print("LOADING PAST SOLUTION FROM FILE: " + warm_start_file)
				return sols[n_buses]

		t = Trajectory(sys_params.T, sys_params.dt, {
			"voltage": n_buses, "current": n_buses, "power": n_buses,
			"delta": n_gens, "omega": n_gens, "Tm": n_gens, "Pc": n_gens,
		}, dtype=np.complex128)
		t.set_constant(["voltage"], list(ic['voltage']))
		t.set_constant(["current"], list(ic['current']))
		t.set_constant(["delta"],   list(ic['delta']))
		t.set_constant(["omega"],   list(ic['omega']))
		t.set_constant(["Tm"],      list(ic['Tm']))
		t.set_constant(["power"],   list(ic['power']))
		t.set_constant(["Pc"],      list(ic['Pc']))
		t.w["voltage"][:, sys_params.N_step:] = ic_post['voltage'].reshape(-1, 1)
		t.w["current"][:, sys_params.N_step:] = ic_post['current'].reshape(-1, 1)
		t.w["power"][:, sys_params.N_step:]   = ic_post['power'].reshape(-1, 1)
		t.w["delta"][:,  sys_params.N_step:]  = ic_post['delta'].reshape(-1, 1)
		t.w["omega"][:,  sys_params.N_step:]  = ic_post['omega'].reshape(-1, 1)
		t.w["Tm"][:,     sys_params.N_step:]  = ic_post['Tm'].reshape(-1, 1)
		t.w["Pc"][:,     sys_params.N_step:]  = ic_post['Pc'].reshape(-1, 1)
		return t

	return dict(
		sys_params=sys_params,
		gens=gens,
		loads=loads,
		make_initial_traj=make_initial_traj,
		Ybus=Ybus,
		gen_costs=sys_params.gen_costs,
		P_min=P_min,
		P_max=P_max,
		V_max=sys_params.V_max,
		weights=weights,
	)


def compute_and_save_precondition_weights(n_buses: int = 24, n_iterations: int = 50, avg_last: int = 10, path: str = "precondition_weights.pkl") -> dict:
	"""
	Calibrates per-variable diagonal preconditioning weights from data instead of guessing.

	Runs ADMM for a fixed n_iterations with flat (all-1.0) weights, records the per-variable
	primal gap ||x.w[v] - z.w[v]|| at each iteration, and averages it over the last avg_last
	iterations (skipping the initial transient) to get a steady-state sense of how far each
	block typically sits from consensus under uniform weighting. Variables with a larger
	residual are penalized harder (D_v = 1 / avg_residual_v^2), then the weights are
	renormalized so the smallest weight is 1.0, matching the convention in
	SysParams.precondition_weights(). The result is cached to `path` for reuse.
	"""
	flat_weights = {v: 1.0 for v in SysParams(n_buses).precondition_weights()}
	setup = _setup_admm_problem(n_buses, weights=flat_weights)

	sys_params = cast(SysParams, setup['sys_params'])
	Ybus = cast(np.ndarray, setup['Ybus'])
	gen_costs = cast(np.ndarray, setup['gen_costs'])
	P_min = cast(np.ndarray, setup['P_min'])
	P_max = cast(np.ndarray, setup['P_max'])
	V_max = cast(np.ndarray, setup['V_max'])
	gens = cast(List[Generator], setup['gens'])
	loads = cast(List[ConstPowerLoad], setup['loads'])
	initial_traj = cast(Callable[[], Trajectory], setup['make_initial_traj'])()
	omega_s = cast(float, sys_params.omega_s)

	Bi = BusBehavioursParallel(gens, loads)
	obj = Objective(Ybus, gen_costs, P_min, P_max, V_max, initial_traj, omega_s=omega_s, omega_band=sys_params.omega_band, weights=flat_weights)

	per_iter_residuals: list[dict[str, float]] = []

	def cb(_iteration, x, z, _u, _r, _s):
		per_iter_residuals.append({var_name: float(np.linalg.norm(x.w[var_name] - z.w[var_name])) for var_name in x.vars})

	print(f"\nCalibrating preconditioning weights over {n_iterations} iterations (flat weights, {n_buses}-bus)...")
	admm(obj, Bi, initial_traj, rho=rho_heuristic, threshold=0.0, max_iterations=n_iterations, callback=cb)

	eps = 1e-8
	avg_residual = {
		var_name: float(np.mean([r[var_name] for r in per_iter_residuals[-avg_last:]]))
		for var_name in flat_weights
	}
	raw_weights = {var_name: 1.0 / max(avg_residual[var_name], eps) ** 2 for var_name in flat_weights}
	min_weight = min(raw_weights.values())
	weights = {var_name: w / min_weight for var_name, w in raw_weights.items()}

	print(f"{'Variable':<10}{'Avg residual':>15}{'Weight':>15}")
	for var_name in weights:
		print(f"{var_name:<10}{avg_residual[var_name]:>15.4e}{weights[var_name]:>15.4f}")

	with open(path, "wb") as f:
		pickle.dump(weights, f)
	print(f"Saved preconditioning weights to {path}")

	return weights


def _resolve_weights(n_buses: int, use_preconditioning_weights: bool, path: str = "precondition_weights.pkl") -> dict:
	"""
	Resolves which preconditioning weights dict to use for a run. If
	use_preconditioning_weights is True, loads the calibrated weights cached by
	compute_and_save_precondition_weights() from `path` (raising a clear error if it
	hasn't been computed yet). Otherwise returns flat (all-1.0) weights, i.e. the
	uniform-rho baseline.
	"""
	flat_weights = {v: 1.0 for v in SysParams(n_buses).precondition_weights()}
	if not use_preconditioning_weights:
		return flat_weights
	if not os.path.exists(path):
		raise FileNotFoundError(
			f"use_preconditioning_weights=True but '{path}' does not exist. "
			f"Run compute_and_save_precondition_weights(n_buses={n_buses}) first to calibrate weights."
		)
	with open(path, "rb") as f:
		return pickle.load(f)


def admm_threshold_sweep(n_buses: int = 24, n_thresholds: int = 5, use_preconditioning_weights: bool = False):
	"""Run ADMM with logarithmically-spaced thresholds in [1e-3, 1e-2] and overlay the
	resulting time-domain dynamics on a single figure to reveal any visual differences."""
	weights = _resolve_weights(n_buses, use_preconditioning_weights)
	setup = _setup_admm_problem(n_buses, weights=weights)
	sys_params        = cast(SysParams, setup['sys_params'])
	gens              = cast(list, setup['gens'])
	loads             = cast(list, setup['loads'])
	make_initial_traj = cast(Callable, setup['make_initial_traj'])
	Ybus                 = cast(np.ndarray, setup['Ybus'])
	gen_costs            = cast(np.ndarray, setup['gen_costs'])
	P_min                = cast(np.ndarray, setup['P_min'])
	P_max                = cast(np.ndarray, setup['P_max'])
	V_max                = cast(np.ndarray, setup['V_max'])
	n_gens               = cast(int, sys_params.n_gens)
	omega_s              = cast(float, sys_params.omega_s)

	thresholds = np.logspace(-3, -2, n_thresholds)
	colors = plt.get_cmap('viridis')(np.linspace(0.1, 0.9, n_thresholds))

	fig, axs = plt.subplots(2, 2, figsize=(14, 10))
	fig.suptitle(f"ADMM Threshold Sweep ({n_buses}-bus, thresholds {thresholds[0]:.1e}–{thresholds[-1]:.1e})")

	for idx, threshold in enumerate(thresholds):
		initial_traj = make_initial_traj()
		Bi = BusBehavioursParallel(gens, loads)
		obj = Objective(Ybus, gen_costs, P_min, P_max, V_max, initial_traj, omega_s=omega_s, omega_band=sys_params.omega_band, weights=weights)
		print(f"\nRunning ADMM with threshold={threshold:.2e}...")
		result = admm(obj, Bi, initial_traj, rho=rho_heuristic, threshold=threshold, max_iterations=2000, weights=weights)

		t_vec = np.arange(result.N) * result.dt
		color = colors[idx]
		label = f"ε={threshold:.2e}"

		for i in range(n_gens):
			lbl = label if i == 0 else None
			axs[0, 0].plot(t_vec, np.real(result.w["omega"][i, :]), color=color, alpha=0.8, label=lbl)
			axs[0, 1].plot(t_vec, np.degrees(np.real(result.w["delta"][i, :])), color=color, alpha=0.8, label=lbl)
			axs[1, 0].plot(t_vec, np.real(result.w["Tm"][i, :]), color=color, alpha=0.8, label=lbl)

		V_mag = np.abs(result.w["voltage"])
		for bus in range(sys_params.n_buses):
			lbl = label if bus == 0 else None
			axs[1, 1].plot(t_vec, V_mag[bus, :], color=color, alpha=0.6, label=lbl)

	axs[0, 0].axhline(omega_s, color='k', linestyle='--', linewidth=0.8, label="ωs")
	axs[0, 0].set_ylabel("ω (rad/s)")
	axs[0, 0].set_title("Rotor Speed")
	axs[0, 0].set_ylim(omega_s - 0.1, omega_s + 0.1)

	axs[0, 1].set_ylabel("δ (deg)")
	axs[0, 1].set_title("Rotor Angle")

	axs[1, 0].set_ylabel("Tm (pu)")
	axs[1, 0].set_title("Mechanical Torque")

	axs[1, 1].set_ylabel("|V| (pu)")
	axs[1, 1].set_title("Bus Voltage Magnitudes")

	for ax in axs.flat:
		ax.set_xlabel("Time (s)")
		ax.axvline(sys_params.N_step * sys_params.dt, color='r', linestyle=':', linewidth=0.8)
		ax.grid(True)
		ax.legend(fontsize=7)

	plt.tight_layout()
	plt.show()


def admm_test(n_buses: int = 24, seq_and_parallel=True, warm_start_file=None, max_iterations=1000, threshold=1e-3, use_preconditioning_weights: bool = False):
	weights = _resolve_weights(n_buses, use_preconditioning_weights)
	setup = _setup_admm_problem(n_buses, weights=weights)
	sys_params = cast(SysParams, setup['sys_params'])
	gens = cast(List[Generator], setup['gens'])
	loads = cast(List[ConstPowerLoad], setup['loads'])
	make_initial_traj = cast(Callable[[], Trajectory], setup['make_initial_traj'])
	Ybus = cast(np.ndarray, setup['Ybus'])
	gen_costs = cast(np.ndarray, setup['gen_costs'])
	P_min = cast(np.ndarray, setup['P_min'])
	P_max = cast(np.ndarray, setup['P_max'])
	V_max = cast(np.ndarray, setup['V_max'])
	n_buses = cast(int, sys_params.n_buses)
	n_gens = cast(int, sys_params.n_gens)
	n_loads = cast(int, sys_params.n_loads)
	omega_s = cast(float, sys_params.omega_s)

	log = logging.getLogger("admm_benchmark")
	log.setLevel(logging.INFO)
	fh = logging.FileHandler("admm_benchmark.log", mode="a")
	fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
	log.addHandler(fh)

	timing_results = {}

	def make_cb(primal_residuals, dual_residuals):
		def cb(iteration, x, z, _u, r, s):
			primal_residuals.append(r)
			dual_residuals.append(s)
			print(f"ADMM iteration {iteration+1}: primal residual = {primal_residuals[-1]:.4e}, dual residual = {dual_residuals[-1]:.4e}")
		return cb

	log.info("Number of Buses: " + str(n_buses))
	test_cases: List[tuple[str, BusBehaviours]] = []
	if seq_and_parallel:
		test_cases.append(("BusBehaviours (sequential)", BusBehavioursSerial(gens, loads)))
	test_cases.append(("BusBehavioursParallel (threaded)", BusBehavioursParallel(gens, loads)))
	
	for label, Bi in test_cases:
		initial_traj = make_initial_traj(warm_start_file=warm_start_file) # type: ignore
		Bi.print_residuals(initial_traj)
		# ress = Bi.compute_residuals(initial_traj) # type: ignore
		# # Plot gen_0_omega
		# plt.plot(ress["gen_0_omega"])
		# plt.show()
		obj = Objective(Ybus, gen_costs, P_min, P_max, V_max, initial_traj, omega_s=omega_s, omega_band=sys_params.omega_band, weights=weights)
		primal_residuals = []
		dual_residuals = []
		print(f"\nRunning ADMM with {label}...")
		t0 = time.perf_counter()
		result = admm(obj, Bi, initial_traj, rho=rho_heuristic, threshold=threshold, max_iterations=max_iterations, callback=make_cb(primal_residuals, dual_residuals), weights=weights) # type: ignore
		elapsed = time.perf_counter() - t0
		timing_results[label] = {"time": elapsed, "iterations": len(primal_residuals), "result": result, "primal_residuals": primal_residuals, "dual_residuals": dual_residuals}
		log.info(f"{label}: {elapsed:.3f}s over {len(primal_residuals)} iteration(s), final primal residual = {primal_residuals[-1]:.4e}, final dual residual = {dual_residuals[-1]:.4e}")
		print(f"  -> {elapsed:.3f}s")
		print(f"  -> {len(primal_residuals)} iterations")
		print(f"  -> final primal residual = {primal_residuals[-1]:.4e}")
		print(f"  -> final dual residual = {dual_residuals[-1]:.4e}")
		print(f"  -> cost = {compute_cost(result, sys_params):.4e}")

	if seq_and_parallel:
		log.info(
			f"Speedup (sequential / parallel): "
			f"{timing_results['BusBehaviours (sequential)']['time'] / timing_results['BusBehavioursParallel (threaded)']['time']:.2f}x"
		)
	print(f"\nBenchmark results logged to admm_benchmark.log")

	# Select solution from sequential test if available, otherwise from parallel test
	if seq_and_parallel and "BusBehaviours (sequential)" in timing_results:
		sol = timing_results["BusBehaviours (sequential)"]["result"]
		primal_residuals = timing_results["BusBehaviours (sequential)"]["primal_residuals"]
		dual_residuals = timing_results["BusBehaviours (sequential)"]["dual_residuals"]
	else:
		sol = timing_results["BusBehavioursParallel (threaded)"]["result"]
		primal_residuals = timing_results["BusBehavioursParallel (threaded)"]["primal_residuals"]
		dual_residuals = timing_results["BusBehavioursParallel (threaded)"]["dual_residuals"]


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
	fig1.suptitle(f"Generator Dynamics ({n_buses}-bus, {n_gens} gens)")

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
	fig2.suptitle(f"Bus Power ({n_buses}-bus)")

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
	fig3.suptitle(f"ADMM Convergence and Feasibility ({n_buses}-bus)")

	axs3[0].semilogy(primal_residuals, marker='o', markersize=3, label="primal residual")
	axs3[0].semilogy(dual_residuals, marker='o', markersize=3, label="dual residual")
	axs3[0].axhline(1e-3, color='r', linestyle='--', label="threshold")
	axs3[0].set_xlabel("Iteration")
	axs3[0].set_ylabel("Residual")
	axs3[0].set_title("Primal and Dual Residuals over Iterations")
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

	# Fig 4: Bus Behaviour residuals. Plot the residuals for each generator together
	fig4, axs4 = plt.subplots(2, 2, figsize=(14, 10))
	fig4.suptitle(f"Bus Behaviour Residuals ({n_buses}-bus)")
	
	bus_res = Bi.compute_residuals(sol) # type: ignore
	
	# Plot generator dynamic residuals
	for i in range(n_gens):
		axs4[0, 0].semilogy(t_vec[1:], np.abs(bus_res[f"gen_{i}_delta"]), label=f"Gen {i+1}")
		axs4[0, 1].semilogy(t_vec[1:], np.abs(bus_res[f"gen_{i}_omega"]), label=f"Gen {i+1}")
		axs4[1, 0].semilogy(t_vec[1:], np.abs(bus_res[f"gen_{i}_Tm"]), label=f"Gen {i+1}")
	
	# Plot algebraic residuals (max over all buses)
	alg_res_all = []
	for i in range(n_gens):
		alg_res_all.append(np.abs(bus_res[f"gen_{i}_algebraic"]).ravel())
	for i in range(n_gens, n_buses):
		alg_res_all.append(np.abs(bus_res[f"load_{i}_power"]).ravel())
	
	max_alg_res = np.max(alg_res_all, axis=0)
	axs4[1, 1].semilogy(t_vec, max_alg_res, color='r', label="Max Algebraic Residual")

	axs4[0, 0].set_title("Delta Dynamics Residual")
	axs4[0, 1].set_title("Omega Dynamics Residual")
	axs4[1, 0].set_title("Tm Dynamics Residual")
	axs4[1, 1].set_title("Algebraic/Power Residuals")
	
	for ax in axs4.flat:
		ax.set_xlabel("Time (s)")
		ax.grid(True)
		ax.legend(fontsize=7)
	plt.tight_layout()

	# Figure 5: Pc
	fig5, axs5 = plt.subplots(1, 1, figsize=(10, 6))
	fig5.suptitle(f"Governor Power Setpoint ({n_buses}-bus)")
	for i in range(n_gens):
		axs5.plot(t_vec, np.real(sol.w["Pc"][i, :]), label=f"Gen {i+1}")
	axs5.set_ylabel("Pc (pu)")
	axs5.set_title("Governor Power Setpoint")
	axs5.legend(fontsize=7)
	axs5.grid(True)
	
	plt.show()

	return timing_results, sol




# test_get_set_var_names()
# test_const_load_projection_refactored()
# run_admm_feasibility_test()

if __name__ == "__main__":
	admm_sols_file_name = "admm_sols_flat_ic.pkl"
	admm_times_file_name = "admm_times.pkl"
	times = {}
	if os.path.exists(admm_times_file_name):
		with open(admm_times_file_name, "rb") as f:
			times = pickle.load(f)
	
	sols = {}
	if os.path.exists(admm_sols_file_name):
		with open(admm_sols_file_name, "rb") as f:
			sols = pickle.load(f)

	nbuses = 4

	timing_results, sol = admm_test(
		n_buses=nbuses,
		seq_and_parallel=False,
		warm_start_file=admm_sols_file_name,
		max_iterations=1000,
		threshold=1e-3,
		use_preconditioning_weights=False
	)
	times[nbuses] = timing_results
	sols[nbuses] = sol

	# admm_threshold_sweep(n_buses=nbuses, n_thresholds=5)

	# compute_and_save_precondition_weights(n_buses=nbuses, n_iterations=50)
	

	# Save times dict to file
	with open(admm_times_file_name, "wb") as f:
		pickle.dump(times, f)
	
	# Save Trajectory solution to file
	
	# Save sols dict to file
	with open(admm_sols_file_name, "wb") as f:
		pickle.dump(sols, f)
