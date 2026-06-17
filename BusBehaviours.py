from Trajectory import Trajectory
from Generator import Generator
from ConstPowerLoad import ConstPowerLoad
from Proxable import Proxable
import numpy as np
from abc import ABC, abstractmethod

class BusBehaviours:
	def __init__(self, gens: list[Generator], loads: list[ConstPowerLoad]):
		self.gens = gens
		self.loads = loads
	
	def compute_residuals(self, trajectory: Trajectory):
		"""
		Computes bus behaviour residuals for a given trajectory.
		"""
		residuals = {}
		for i, gen in enumerate(self.gens):
			gen_vars = ["voltage", "current", "delta", "omega", "Tm", "power", "Pc"]
			t = Trajectory(trajectory.T, trajectory.dt, {v: 1 for v in gen_vars}, dtype=trajectory.dtype)
			for v in gen_vars:
				t.w[v] = trajectory.w[v][[i], :]
			res = gen.compute_residual(t)
			for k, v in res.items():
				residuals[f"gen_{i}_{k}"] = v

		for i, load in enumerate(self.loads, start=len(self.gens)):
			load_vars = ["voltage", "current", "power"]
			t = Trajectory(trajectory.T, trajectory.dt, {v: 1 for v in load_vars}, dtype=trajectory.dtype)
			for v in load_vars:
				t.w[v] = trajectory.w[v][[i], :]
			res = load.compute_residual(t)
			for k, v in res.items():
				residuals[f"load_{i}_{k}"] = v
		return residuals
	
	def print_residuals(self, trajectory: Trajectory):
		residuals = self.compute_residuals(trajectory)
		for k, v in residuals.items():
			print(f"{k}: {np.linalg.norm(v)}")


class BusBehavioursSerial(Proxable, BusBehaviours):
	def __init__(self, gens: list[Generator], loads: list[ConstPowerLoad]):
		self.gens = gens
		self.loads = loads

	def prox(self, trajectory: Trajectory, rho=1.0) -> Trajectory:
		ret = trajectory.copy()
		for i, gen in enumerate(self.gens):
			# print(f"Projecting onto generator {i+1}/{len(self.gens)}")
			gen_vars = ["voltage", "current", "delta", "omega", "Tm", "power", "Pc"]
			t = Trajectory(trajectory.T, trajectory.dt, {v: 1 for v in gen_vars}, dtype=trajectory.dtype)
			for v in gen_vars:
				t.w[v] = trajectory.w[v][[i], :]
			projected_t = gen.prox(t, rho)
			for v in gen_vars:
				ret.w[v][[i], :] = projected_t.w[v]
		for i, load in enumerate(self.loads, start=len(self.gens)):
			# print(f"Projecting onto load {i+1}/{len(self.loads)}")
			load_vars = ["voltage", "current", "power"]
			t = Trajectory(trajectory.T, trajectory.dt, {v: 1 for v in load_vars}, dtype=trajectory.dtype)
			for v in load_vars:
				t.w[v] = trajectory.w[v][[i], :]
			projected_t = load.prox(t, rho)
			for v in load_vars:
				ret.w[v][[i], :] = projected_t.w[v]
		return ret

class BusBehavioursParallel(Proxable, BusBehaviours):
	def __init__(self, gens: list[Generator], loads: list[ConstPowerLoad]):
		self.gens = gens
		self.loads = loads

	def prox(self, trajectory: Trajectory, rho=1.0) -> Trajectory:
		ret = trajectory.copy()

		def project_gen(args):
			i, gen = args
			# print(f"Projecting onto generator {i+1}/{len(self.gens)}")
			gen_vars = ["voltage", "current", "delta", "omega", "Tm", "power", "Pc"]
			t = Trajectory(trajectory.T, trajectory.dt, {v: 1 for v in gen_vars}, dtype=trajectory.dtype)
			for v in gen_vars:
				t.w[v] = trajectory.w[v][[i], :]
			projected_t = gen.prox(t)
			return i, gen_vars, projected_t

		def project_load(args):
			i, load = args
			# print(f"Projecting onto load {i+1}/{len(self.loads)}")
			load_vars = ["voltage", "current", "power"]
			t = Trajectory(trajectory.T, trajectory.dt, {v: 1 for v in load_vars}, dtype=trajectory.dtype)
			for v in load_vars:
				t.w[v] = trajectory.w[v][[i], :]
			projected_t = load.prox(t)
			return i, load_vars, projected_t

		from concurrent.futures import ThreadPoolExecutor
		gen_args = list(enumerate(self.gens))
		load_args = [(i + len(self.gens), load) for i, load in enumerate(self.loads)]

		with ThreadPoolExecutor() as executor:
			gen_futures = executor.map(project_gen, gen_args)
			load_futures = executor.map(project_load, load_args)
			for i, vars_, projected_t in gen_futures:
				for v in vars_:
					ret.w[v][[i], :] = projected_t.w[v]
			for i, vars_, projected_t in load_futures:
				for v in vars_:
					ret.w[v][[i], :] = projected_t.w[v]

		return ret