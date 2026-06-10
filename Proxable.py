from abc import ABC, abstractmethod
from Trajectory import Trajectory

class Proxable(ABC):
	"""
	An abstract base class representing a proximal operator for a subclass-defined function. Namely, subclasses should implement:
	
	Prox_{lambda, f}(z) = min_x f(x) + rho/2||x - z||^2

	Methods
	-------
	prox(trajectory: Trajectory) -> Trajectory
		Solves the proximal operator for the given trajectory and returns the resulting trajectory.
	"""

	@abstractmethod
	def prox(self, trajectory: Trajectory, rho: float = 1.0) -> Trajectory:
		...