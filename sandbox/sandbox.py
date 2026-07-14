import numpy as np
from typing import Callable, Dict
import casadi as ca
from abc import ABC, abstractmethod

# class Proxable(ABC):
#     """
#     Abstract class implementing the solution to
#     Prox_{rho, f}(z) = min_x f(x) + rho/2||x - z||^2
#     """
#     @abstractmethod
#     def prox(self, z: np.ndarray, rho: float) -> np.ndarray:
#         pass

# def admm(f: Proxable,
#          g: Proxable,
#          z0: np.ndarray,
#          rho=lambda i, prev, r, s: 2.0,
#          threshold=1e-3,
#          max_iterations=1000,
#          callback=None):
#     """
#     Minimizes a constrained optimization problem using the Alternating Direction Method of Multipliers (ADMM).

#     Parameters
#     ----------
#     f : Proxable
#         The (possibly constrained) objective function to be minimized.
#     g : Proxable
#         The projection operator representing the constraints.
#     z0 : np.ndarray
#         The initial guess for the solution.
#     callback : callable, optional 
#         Called as callback(iteration, x, z, u) at the end of each iteration.
#     """

#     # Initialize x0, z0, mu0
#     zs: list[np.ndarray] = [z0.copy()]
#     xs: list[np.ndarray] = [np.zeros_like(z0)]
#     us: list[np.ndarray] = [np.zeros_like(z0)]

#     rs = [np.linalg.norm(xs[-1] - zs[-1])]
#     ss = [np.linalg.norm(xs[-1] - zs[-1])]
#     rhos = [2.0]

#     for iteration in range(max_iterations-1):
#         new_rho = rho(iteration, rhos[-1], rs[-1], ss[-1])
#         rhos.append(new_rho)
#         # Rescale u when rho changes to keep λ = rho*u continuous
#         if new_rho != rhos[-2]:
#             us[-1] = us[-1] * (rhos[-2] / new_rho)
            
#         xs.append(f.prox(zs[-1] - us[-1], rhos[-1]))
#         zs.append(g.prox(xs[-1] + us[-1], rhos[-1]))
#         us.append(us[-1] + (xs[-1] - zs[-1]))


#         rs.append(np.linalg.norm(xs[-1] - zs[-1]))
#         ss.append(rhos[-1] * np.linalg.norm(zs[-1] - zs[-2]))

#         if callback is not None:
#             callback(iteration, xs[-1], zs[-1], us[-1], rs[-1], ss[-1])

#         if rs[-1] < threshold and ss[-1] < threshold:
#             break

#     return xs[-1]


def make_config():
    config = {}
    n_buses = 3
    n_gens = 2
    
    Y_bus = np.array([[ 4.902-19.608j, -2.941+11.765j, -1.961 +7.843j],
                      [-2.941+11.765j,  5.294-21.176j, -2.353 +9.412j],
                      [-1.961 +7.843j, -2.353 +9.412j,  4.314-17.255j]])

    config["n_buses"] = n_buses
    config["n_gens"] = n_gens
    config["G"] = np.real(Y_bus)
    config["B"] = np.imag(Y_bus)
    config["costs"] = np.array([1.0, 1.2])
    config["load_P"] = np.array([1.0])
    config["load_Q"] = np.array([0.5])
    config["V_max"] = np.array([1.1, 1.1, 1.1])
    config["V_min"] = np.array([0.9, 0.9, 0.9])
    config["P_max"] = np.array([1.2, 1.2])
    config["P_min"] = np.array([0.2, 0.2])
    config["Q_max"] = np.array([1.3, 1.3])
    config["Q_min"] = np.array([-0.2, -0.2])

    return config

def solve_opf_centralized(config: Dict):
    G = config["G"]
    B = config["B"]
    n_buses = config["n_buses"]
    n_gens = config["n_gens"]
    costs = config["costs"]
    load_P = config["load_P"]
    load_Q = config["load_Q"]
    V_max = config["V_max"]
    V_min = config["V_min"]
    P_max = config["P_max"]
    P_min = config["P_min"]
    Q_max = config["Q_max"]
    Q_min = config["Q_min"]

    opti = ca.Opti()

    V_re = opti.variable(n_buses)
    V_im = opti.variable(n_buses)
    I_re = opti.variable(n_buses)
    I_im = opti.variable(n_buses)
    P = opti.variable(n_buses)
    Q = opti.variable(n_buses)

    opti.minimize(sum(costs[i] * P[i]**2 for i in range(n_gens)))

    # Power flow constraints
    opti.subject_to(I_re == G @ V_re - B @ V_im)
    opti.subject_to(I_im == B @ V_re + G @ V_im)

    for i in range(n_buses):
        opti.subject_to(P[i] == V_re[i] * I_re[i] + V_im[i] * I_im[i])
        opti.subject_to(Q[i] == V_im[i] * I_re[i] - V_re[i] * I_im[i])
    
    for i in range(n_gens, n_buses):
        opti.subject_to(P[i] == -load_P[i - n_gens])
        opti.subject_to(Q[i] == -load_Q[i - n_gens])
    
    # Reference (slack) bus angle
    opti.subject_to(V_im[0] == 0)

    # Operational constraints
    for i in range(n_buses):
        V_sq = V_re[i]**2 + V_im[i]**2
        opti.subject_to(V_sq >= V_min[i])
        opti.subject_to(V_sq <= V_max[i])
    
    for i in range(n_gens):
        opti.subject_to(P[i] >= P_min[i])
        opti.subject_to(P[i] <= P_max[i])
        opti.subject_to(Q[i] >= Q_min[i])
        opti.subject_to(Q[i] <= Q_max[i])

    opti.set_initial(V_re, np.ones(n_buses))
    opti.set_initial(V_im, np.zeros(n_buses))

    opti.solver('ipopt')

    sol = opti.solve()

    print("V_re:", sol.value(V_re))
    print("V_im:", sol.value(V_im))
    print("I_re:", sol.value(I_re))
    print("I_im:", sol.value(I_im))
    print("P:", sol.value(P))
    print("Q:", sol.value(Q))

    return sol

if __name__ == "__main__":
    config = make_config()
    solve_opf_centralized(config)