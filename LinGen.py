from Trajectory import Trajectory
from Projectable import Projectable
from SysParams import SysParams
import numpy as np
import scipy.linalg
import cvxpy as cp

class LinGen(Projectable):
    def __init__(self, sys_params: SysParams, trajectory: Trajectory, use_constraints=True):
        """
        Initializes the linearized generator model.
        Produces matrices F, G, H for the discrete-time linear system:
        [delta_{t+1}, omega_{t+1}, Pm_{t+1}]^T = F * [delta_t, omega_t, Pm_t]^T + G * [theta_t, Pc_t]^T + H
        """
        self.trajectory = trajectory
        self.sys_params = sys_params
        self.use_constraints = use_constraints

        # Extract parameters
        M = sys_params.M
        Xd = sys_params.Xd
        D = sys_params.D
        omega_s = sys_params.omega_s
        R = sys_params.Rd
        Tsv = sys_params.Tsv

        A_cont = np.array([[0, 1, 0],
                             [-1/(Xd*M), -D/(M*omega_s), 1/M],
                             [0, -1/(R*Tsv*omega_s), -1/Tsv]])

        B_cont= np.array([[0, 0],
                           [1/(Xd*M), 0],
                           [0, 1/Tsv]])
        
        E_cont = np.array([[-omega_s],
                           [D/M],
                           [1/(R*Tsv)]])
        
        M_mat = np.vstack((np.hstack((A_cont, B_cont, E_cont)), np.zeros((2, 5))))

        exp_M = scipy.linalg.expm(M_mat * trajectory.dt)
        self.F = exp_M[:3, :3]
        self.G = exp_M[:3, 3:5]
        self.H = exp_M[:3, 5:]

        if not use_constraints:
            # Build projection matrices for unconstrained case
            # TODO: Consider sparse Cholesky factorization for larger time horizons
            # TODO: Build matrices
            self.projector = 1

    def project(self, trajectory: Trajectory) -> Trajectory:
        if self.use_constraints:
            return self.project_with_constraints(trajectory)
        else:
            return self.project_without_constraints(trajectory)
    
    def project_without_constraints(self, trajectory: Trajectory) -> Trajectory:
        """
        Projects the given trajectory onto the linearized generator model without constraints.
        Parameters
        ----------
        trajectory : Trajectory
            The trajectory to project, given as [delta, omega, Pm, theta, Pc] over time
        
        Returns
        -------
        Trajectory
            The projected trajectory.
        """
        return trajectory.copy() * self.projector
    
    def project_with_constraints(self, trajectory: Trajectory) -> Trajectory:
        """
        Projects the given trajectory onto the linearized generator model with constraints.
        Parameters
        ----------
        trajectory : Trajectory
            The trajectory to project, given as [delta, omega, Pm, theta, Pc] over time
        
        Returns
        -------
        Trajectory
            The projected trajectory.
        """

        # Extract the relevant states and inputs from the trajectory
        delta = trajectory.w['delta']
        omega = trajectory.w['omega']
        Pm = trajectory.w['Pm']
        theta = trajectory.w['theta']
        Pc = trajectory.w['Pc']

        # Build optimization variables
        delta_var = cp.Variable((1, trajectory.N))
        omega_var = cp.Variable((1, trajectory.N))
        Pm_var = cp.Variable((1, trajectory.N))
        theta_var = cp.Variable((1, trajectory.N))
        Pc_var = cp.Variable((1, trajectory.N))
        
        # Dynamic Constraints based on the linearized model
        constraints = []
        for t in range(trajectory.N - 1):
            constraints += [
                cp.hstack([delta_var[:, t+1], omega_var[:, t+1], Pm_var[:, t+1]]) ==
                self.F @ cp.hstack([delta_var[:, t], omega_var[:, t], Pm_var[:, t]]) +
                self.G @ cp.hstack([theta_var[:, t], Pc_var[:, t]]) +
                self.H
            ]
        
        # Operational constraints (e.g., limits on power, frequency)
        constraints += [
            Pm_var >= self.sys_params.P_min,
            Pm_var <= self.sys_params.P_max
        ]

        # Objective: minimize deviation from original trajectory
        objective = cp.Minimize(
            cp.norm(delta - delta_var, 'fro') +
            cp.norm(omega - omega_var, 'fro') +
            cp.norm(Pm - Pm_var, 'fro') +
            cp.norm(theta - theta_var, 'fro') +
            cp.norm(Pc - Pc_var, 'fro')
        )

        # Solve the optimization problem
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        # Construct the projected trajectory
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise ValueError("Optimization problem did not solve to optimality.")
        if prob.status == cp.OPTIMAL_INACCURATE:
            print("Warning: Optimization problem solved to optimality but with numerical issues.")
        projected_trajectory = Trajectory(trajectory.T, trajectory.dt, trajectory.vars)
        
        projected_trajectory.w['delta'] = delta_var.value if delta_var.value is not None else delta
        projected_trajectory.w['omega'] = omega_var.value if omega_var.value is not None else omega
        projected_trajectory.w['Pm'] = Pm_var.value if Pm_var.value is not None else Pm
        projected_trajectory.w['theta'] = theta_var.value if theta_var.value is not None else theta
        projected_trajectory.w['Pc'] = Pc_var.value if Pc_var.value is not None else Pc

        return projected_trajectory
    

