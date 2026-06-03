import numpy as np

class SysParams:
	"""
	System parameters class that encapsulates network and generator parameters.
	Extracted from admm_test to enable reusable OPF initialization.
	"""
	def __init__(self, n_buses: int):
		self.omega_s = 2*np.pi*60
		self.n_buses = n_buses
		self.n_gens = n_buses // 2
		self.n_loads = n_buses // 2

		# Simulation parameters
		self.T = 5.0
		self.dt = 0.05
		self.N = int(self.T / self.dt)

		# Generator parameters
		self.H = 8
		self.D = 2.0
		self.Rd = 0.04
		self.X_p = 0.0608
		self.Tsv = 2

		# Power dispatch targets
		self.S_gen0  = 0.40 + 0.08j
		self.S_load0 = -0.40 - 0.08j

		# Constraints
		self.P_min = np.full(self.n_gens, 0.1)
		self.P_max = np.full(self.n_gens, 2.0)
		self.V_max = np.concatenate([
			np.full(self.n_gens, 2.2),
			np.full(self.n_loads, 2.15)
		])
		self.gen_costs = np.ones(self.n_gens)

		# Build Ybus matrix with two rings + cross-links topology
		self.Ybus = self._build_ybus()

	def _build_ybus(self) -> np.ndarray:
		"""Build Ybus matrix: two rings (gen/load) + cross-links."""
		half = self.n_gens // 2
		branches = (
			[(i, (i+1) % self.n_gens) for i in range(self.n_gens)] +
			[(self.n_gens + i, self.n_gens + (i+1) % self.n_loads) for i in range(self.n_loads)] +
			[(i, self.n_gens + (2*i) % self.n_loads) for i in range(half)] +
			[(half + i, self.n_gens + (2*i+1) % self.n_loads) for i in range(half)]
		)
		y_line = 1.0 / (0.01 + 0.085j)
		ysh_line = 0.044j
		Ybus = np.zeros((self.n_buses, self.n_buses), dtype=complex)
		for bi, bj in branches:
			Ybus[bi, bi] += y_line + ysh_line
			Ybus[bj, bj] += y_line + ysh_line
			Ybus[bi, bj] -= y_line
			Ybus[bj, bi] -= y_line
		return Ybus