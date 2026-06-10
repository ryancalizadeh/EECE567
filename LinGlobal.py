from Solvable import Solvable
from Trajectory import Trajectory
from SysParams import SysParams
import cvxpy as cp
import numpy as np

class LinGlobal(Solvable):
    """
    A class reprsenting the minimization of generation cost subject to operational constraints and network behaviour
    trajectory = [delta, omega, Pm, theta, Pc]
    """

    def __init__(self, sysParams: SysParams, trajectory: Trajectory, rho=1.0):
        self.rho = rho
        self.traj = trajectory
        self.sysParams = sysParams

        P_min = np.array(sysParams.P_min).reshape(-1, 1)
        P_max = np.array(sysParams.P_max).reshape(-1, 1)

        self.delta = cp.Variable((sysParams.n_gens, trajectory.N))
        self.omega = cp.Variable((sysParams.n_gens, trajectory.N))
        self.Pm = cp.Variable((sysParams.n_gens, trajectory.N))
        self.theta = cp.Variable((sysParams.n_buses, trajectory.N))
        self.Pc = cp.Variable((sysParams.n_gens, trajectory.N))

        self.deltaw = cp.Parameter(self.delta.shape)
        self.omegaw = cp.Parameter(self.omega.shape)
        self.Pmw = cp.Parameter(self.Pm.shape)
        self.thetaw = cp.Parameter(self.theta.shape)
        self.Pcw = cp.Parameter(self.Pc.shape)

        self.x = cp.vstack([self.delta, self.omega, self.Pm, self.theta, self.Pc])
        self.w = cp.vstack([self.deltaw, self.omegaw, self.Pmw, self.thetaw, self.Pcw])

        # Self.cost = sum over electrical power of each generator and each timestep of Pe_i^2 * gen_costs[i]
        self.cost = cp.sum([cp.quad_over_lin((self.delta[i, :] - self.theta[i, :]) / sysParams.Xd, 1/sysParams.gen_costs[i]) for i in range(sysParams.n_gens)])

        # Self.penalty = rho * ||x-w||_2^2, recalling that these are complex numbers
        self.penalty = self.rho * cp.sum_squares(self.x - self.w)


        self.constraints = [(self.delta - self.theta) / sysParams.Xd == np.imag(sysParams.Ybus) @ self.theta]
        # TODO recall that this is dynamic, so as written this won't work


        # TODO Add line current limits

        self.problem = cp.Problem(cp.Minimize(self.cost + self.penalty), self.constraints)

    def solve(self, t: Trajectory) -> Trajectory:
        # self.deltaw.value = t.get_var_names(["voltage"])
        # ...

        self.problem.solve()

        # Extract solution and return a trajectory
        ret = t.copy()

        # if self.V.value is not None:
        #     ret.set_var_names(["voltage"], self.V.value)
        # if self.I.value is not None:
        #     ret.set_var_names(["current"], self.I.value)
        # if self.S.value is not None:
        #     ret.w["power"][:self.g, :] = self.S.value
        # if self.Pc_var.value is not None:
        #     ret.w["Pc"][:self.g, :] = self.Pc_var.value

        if self.problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"Optimization failed with status {self.problem.status}")

        return ret
