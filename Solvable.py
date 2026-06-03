from abc import ABC, abstractmethod
from Trajectory import Trajectory

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