from abc import ABC, abstractmethod
from Trajectory import Trajectory

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