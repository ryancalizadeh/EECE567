import numpy as np

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
	
	def __mul__(self, scalar):
		result = Trajectory(self.T, self.dt, self.vars)
		for var_name in self.vars:
			result.w[var_name] = self.w[var_name] * scalar
		return result

	def __rmul__(self, scalar):
		return self.__mul__(scalar)

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
