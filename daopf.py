import os

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import time
import logging
from saap import Trajectory, Generator, ConstPowerLoad, SysParams
from OPF import solve_opf, ic_from_opf
from scipy.linalg import null_space


def solve_daopf(
    gens: list,
    loads: list,
    load_buses: list,
    sys_params: SysParams,
    ic: dict[str, np.ndarray],
    x0_override: np.ndarray | None = None,
    opts_override: dict | None = None,
) -> tuple:
    """
    Solves the dynamics-aware OPF as a single monolithic NLP using CasADI/IPOPT.

    Decision variables (real-valued):
      V_re, V_im  : (n_buses, N)  bus voltages
      I_re, I_im  : (n_buses, N)  bus currents
      delta_i     : (N,)          rotor angle deviation per generator
      omega_i     : (N,)          rotor speed per generator
      Tm_i        : (N,)          mechanical torque per generator
    """
    P_min = sys_params.P_min
    P_max = sys_params.P_max
    V_max = sys_params.V_max
    gen_costs = sys_params.gen_costs
    n_buses = sys_params.n_buses
    
    T = sys_params.T
    dt = sys_params.dt
    N = sys_params.N
    n_gens = sys_params.n_gens
    n_loads = sys_params.n_loads
    omega_s = sys_params.omega_s
    Ybus = sys_params.Ybus

    Ybus_re = np.real(Ybus)
    Ybus_im = np.imag(Ybus)

    # ------------------------------------------------------------------ #
    # Decision variables
    # ------------------------------------------------------------------ #
    V_re = ca.MX.sym("V_re", n_buses, N) # type: ignore
    V_im = ca.MX.sym("V_im", n_buses, N) # type: ignore
    I_re = ca.MX.sym("I_re", n_buses, N) # type: ignore
    I_im = ca.MX.sym("I_im", n_buses, N) # type: ignore

    # Per-generator state variables
    deltas = [ca.MX.sym(f"delta_{i}", N) for i in range(n_gens)] # type: ignore
    omegas = [ca.MX.sym(f"omega_{i}", N) for i in range(n_gens)] # type: ignore
    Tms    = [ca.MX.sym(f"Tm_{i}",    N) for i in range(n_gens)] # type: ignore
    # Free control: governor power setpoint (optimizer chooses this trajectory)
    Pcs    = [ca.MX.sym(f"Pc_{i}",    N) for i in range(n_gens)] # type: ignore

    # Flatten to one vector for nlpsol
    x_parts = [
        ca.reshape(V_re, -1, 1),
        ca.reshape(V_im, -1, 1),
        ca.reshape(I_re, -1, 1),
        ca.reshape(I_im, -1, 1),
    ]
    for i in range(n_gens):
        x_parts += [deltas[i], omegas[i], Tms[i], Pcs[i]]
    x = ca.vertcat(*x_parts)

    # ------------------------------------------------------------------ #
    # Bounds
    # ------------------------------------------------------------------ #
    lbx_V_re = np.full((n_buses, N), -np.inf)
    ubx_V_re = np.full((n_buses, N),  np.inf)
    lbx_V_im = np.full((n_buses, N), -np.inf)
    ubx_V_im = np.full((n_buses, N),  np.inf)
    lbx_I_re = np.full((n_buses, N), -np.inf)
    ubx_I_re = np.full((n_buses, N),  np.inf)
    lbx_I_im = np.full((n_buses, N), -np.inf)
    ubx_I_im = np.full((n_buses, N),  np.inf)

    P_min_arr = np.array(P_min)
    P_max_arr = np.array(P_max)

    lbx_delta = [np.full(N, -np.inf) for _ in range(n_gens)]
    ubx_delta = [np.full(N,  np.inf) for _ in range(n_gens)]
    lbx_omega = [np.full(N, omega_s - 0.08) for _ in range(n_gens)]
    ubx_omega = [np.full(N, omega_s + 0.08) for _ in range(n_gens)]
    lbx_Tm    = [np.full(N, -np.inf) for _ in range(n_gens)]
    ubx_Tm    = [np.full(N,  np.inf) for _ in range(n_gens)]
    lbx_Pc    = [np.full(N, P_min_arr[i]) for i in range(n_gens)]
    ubx_Pc    = [np.full(N, P_max_arr[i]) for i in range(n_gens)]
    # TODO: Fix bounds on Pc, should be on Tm

    lbx_parts = [lbx_V_re.flatten(), lbx_V_im.flatten(),
                 lbx_I_re.flatten(), lbx_I_im.flatten()]
    ubx_parts = [ubx_V_re.flatten(), ubx_V_im.flatten(),
                 ubx_I_re.flatten(), ubx_I_im.flatten()]
    for i in range(n_gens):
        lbx_parts += [lbx_delta[i], lbx_omega[i], lbx_Tm[i], lbx_Pc[i]]
        ubx_parts += [ubx_delta[i], ubx_omega[i], ubx_Tm[i], ubx_Pc[i]]
    lbx = np.concatenate(lbx_parts)
    ubx = np.concatenate(ubx_parts)

    # ------------------------------------------------------------------ #
    # Constraints
    # ------------------------------------------------------------------ #
    g_list = []
    lbg_list = []
    ubg_list = []

    def eq(expr):
        g_list.append(ca.reshape(expr, -1, 1))
        sz = expr.numel()
        lbg_list.append(np.zeros(sz))
        ubg_list.append(np.zeros(sz))

    def ineq_lb(expr, lb):
        # lb <= expr  ->  0 <= expr - lb
        g_list.append(ca.reshape(expr - lb, -1, 1))
        sz = expr.numel()
        lbg_list.append(np.zeros(sz))
        ubg_list.append(np.full(sz, np.inf))

    def ineq_ub(expr, ub):
        # expr <= ub  ->  0 <= ub - expr
        g_list.append(ca.reshape(ub - expr, -1, 1))
        sz = expr.numel()
        lbg_list.append(np.zeros(sz))
        ubg_list.append(np.full(sz, np.inf))

    # # 1a. Angle reference: pin absolute phase at bus 0 to remove the global
    # # rotational symmetry (V,I,delta) -> (V*e^{j*theta}, I*e^{j*theta}, delta+theta),
    # # which otherwise leaves the objective and all other constraints invariant.
    # eq(V_im[0, 0])

    # 1. KCL: Ybus @ V = I  (real and imaginary parts, all timesteps)
    for k in range(N):
        kcl_re = Ybus_re @ V_re[:, k] - Ybus_im @ V_im[:, k] - I_re[:, k]
        kcl_im = Ybus_im @ V_re[:, k] + Ybus_re @ V_im[:, k] - I_im[:, k]
        eq(kcl_re)
        eq(kcl_im)

    # 2. Generator DAE constraints (trapezoidal discretization)
    for i, gen in enumerate(gens):
        bus = i  # generator i is connected to bus i
        delta = deltas[i]
        omega = omegas[i]
        Tm    = Tms[i]
        Pc    = Pcs[i]

        # Initial conditions (V/I at t=0 are determined by KCL + algebraic + load)
        eq(delta[0])
        eq(omega[0] - omega_s)
        eq(Tm[0] - gen.Pc)

        # Terminal conditions (omega=omega_s at t=T)
        eq(omega[-1] - omega_s)
        # eq(delta[-1] - delta[-2])
        # eq(omega[-1] - omega[-2])
        # eq(Tm[-1] - Tm[-2])

        # # Terminal governor steady-state: with omega[-1]=omega_s already enforced,
        # # Tm_dot=0 requires Pc[-1] = Tm[-1]. Without this, Pc[-1] is paired with
        # # Pc[-2] by a single ODE equation and is otherwise unconstrained (no cost
        # # or bound references Pc), leaving a flat direction among global optima.
        # eq(-Tm[-1] + Pc[-1] - (1 / gen.R) * (omega[-1] / omega_s - 1))

        # ODE: trapezoidal rule for k = 0..N-2
        for k in range(N - 1):
            # --- delta_dot = omega - omega_s ---
            f_d_k  = omega[k]   - omega_s
            f_d_k1 = omega[k+1] - omega_s
            eq(delta[k+1] - delta[k] - (dt/2) * (f_d_k + f_d_k1))

            # --- omega_dot ---
            E_re_k  = gen.E * ca.cos(delta[k]   + gen.delta0)
            E_im_k  = gen.E * ca.sin(delta[k]   + gen.delta0)
            E_re_k1 = gen.E * ca.cos(delta[k+1] + gen.delta0)
            E_im_k1 = gen.E * ca.sin(delta[k+1] + gen.delta0)

            Pe_k  = E_re_k  * I_re[bus, k]   + E_im_k  * I_im[bus, k]
            Pe_k1 = E_re_k1 * I_re[bus, k+1] + E_im_k1 * I_im[bus, k+1]

            f_o_k  = (Tm[k]   - Pe_k  - gen.D * (omega[k]   / omega_s - 1)) * omega_s / (2 * gen.H)
            f_o_k1 = (Tm[k+1] - Pe_k1 - gen.D * (omega[k+1] / omega_s - 1)) * omega_s / (2 * gen.H)
            eq(omega[k+1] - omega[k] - (dt/2) * (f_o_k + f_o_k1))

            # --- Tm_dot ---
            f_T_k  = (-Tm[k]   + Pc[k]   - (1/gen.R) * (omega[k]   / omega_s - 1)) / gen.Tsv
            f_T_k1 = (-Tm[k+1] + Pc[k+1] - (1/gen.R) * (omega[k+1] / omega_s - 1)) / gen.Tsv
            eq(Tm[k+1] - Tm[k] - (dt/2) * (f_T_k + f_T_k1))

        # Algebraic: E - Zd*I - V = 0 at every timestep
        Zd_re = gen.Zd.real
        Zd_im = gen.Zd.imag
        for k in range(N):
            E_re_k = gen.E * ca.cos(delta[k] + gen.delta0)
            E_im_k = gen.E * ca.sin(delta[k] + gen.delta0)
            alg_re = E_re_k - Zd_re * I_re[bus, k] + Zd_im * I_im[bus, k] - V_re[bus, k]
            alg_im = E_im_k - Zd_re * I_im[bus, k] - Zd_im * I_re[bus, k] - V_im[bus, k]
            eq(alg_re)
            eq(alg_im)

    # 3. Load power constraints: V * conj(I) = S_load(t)
    for load, bus in zip(loads, load_buses):
        for k in range(N):
            s = load.S(k * dt)
            P_load = s.real
            Q_load = s.imag
            # Re(V * conj(I)) = P_load
            eq(V_re[bus, k] * I_re[bus, k] + V_im[bus, k] * I_im[bus, k] - P_load)
            # Im(V * conj(I)) = Q_load
            eq(V_im[bus, k] * I_re[bus, k] - V_re[bus, k] * I_im[bus, k] - Q_load)

    # 4. Generation limits: P_min <= Re(V*conj(I)) <= P_max  per generator bus
    P_min = np.array(P_min)
    P_max = np.array(P_max)
    for i in range(n_gens):
        for k in range(N):
            P_gen = V_re[i, k] * I_re[i, k] + V_im[i, k] * I_im[i, k]
            ineq_lb(P_gen, P_min[i])
            ineq_ub(P_gen, P_max[i])

    # 5. Voltage magnitude limits: |V|^2 <= V_max^2
    V_max = np.array(V_max)
    for bus in range(n_buses):
        for k in range(N):
            V_sq = V_re[bus, k]**2 + V_im[bus, k]**2
            ineq_ub(V_sq, V_max[bus]**2)
    V_min = 0.8  # per-unit minimum voltage magnitude
    for bus in range(n_buses):
        for k in range(N):
            V_sq = V_re[bus, k]**2 + V_im[bus, k]**2
            ineq_lb(V_sq, V_min**2)


    g = ca.vertcat(*g_list)
    lbg = np.concatenate(lbg_list)
    ubg = np.concatenate(ubg_list)

    # ------------------------------------------------------------------ #
    # Objective: minimize generation cost sum_i c_i * P_gen_i(k)^2
    # ------------------------------------------------------------------ #
    obj = 0
    for i, c in enumerate(gen_costs):
        for k in range(N):
            P_gen = V_re[i, k] * I_re[i, k] + V_im[i, k] * I_im[i, k]
            obj += c * P_gen**2

    # ------------------------------------------------------------------ #
    # Warm start: steady-state initial conditions
    # ------------------------------------------------------------------ #
    V_re0 = np.tile(np.real(ic['voltage']).reshape(-1, 1), (1, N))
    V_im0 = np.tile(np.imag(ic['voltage']).reshape(-1, 1), (1, N))
    I_re0 = np.tile(np.real(ic['current']).reshape(-1, 1), (1, N))
    I_im0 = np.tile(np.imag(ic['current']).reshape(-1, 1), (1, N))
    x0_parts = [V_re0.flatten(), V_im0.flatten(), I_re0.flatten(), I_im0.flatten()]
    for i in range(n_gens):
        x0_parts += [
            np.zeros(N),                    # delta deviation: 0 at t=0 per IC constraint
            np.full(N, ic['omega'][i]),
            np.full(N, ic['Tm'][i]),
            np.full(N, ic['Pc'][i]),
        ]
    x0 = np.concatenate(x0_parts)

    # ------------------------------------------------------------------ #
    # Solve
    # ------------------------------------------------------------------ #
    default_opts = {
        "ipopt.print_level": 1,
        "ipopt.tol": 1e-6,
        "ipopt.max_iter": 5000,
        "ipopt.acceptable_tol": 1e-5,
        "ipopt.acceptable_iter": 10,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.nlp_scaling_method": "gradient-based",
    }
    opts = opts_override if opts_override is not None else default_opts
    x0_use = x0_override if x0_override is not None else x0
    nlp = {"f": obj, "x": x, "g": g}
    solver = ca.nlpsol("daopf_solver", "ipopt", nlp, opts)

    print("Solving centralized DA-OPF NLP...")
    sol = solver(x0=x0_use, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    print(f"Solver status: {solver.stats()['return_status']}")

    # ------------------------------------------------------------------ #
    # Extract solution into a Trajectory
    # ------------------------------------------------------------------ #
    xv = sol["x"].full().flatten()
    offset = 0

    def take(shape):
        nonlocal offset
        sz = int(np.prod(shape))
        arr = xv[offset: offset + sz].reshape(shape, order='F')
        offset += sz
        return arr

    sol_V_re = take((n_buses, N))
    sol_V_im = take((n_buses, N))
    sol_I_re = take((n_buses, N))
    sol_I_im = take((n_buses, N))
    sol_delta = np.zeros((n_gens, N))
    sol_omega = np.zeros((n_gens, N))
    sol_Tm    = np.zeros((n_gens, N))
    sol_Pc    = np.zeros((n_gens, N))
    for i in range(n_gens):
        sol_delta[i] = take((N,))
        sol_omega[i] = take((N,))
        sol_Tm[i]    = take((N,))
        sol_Pc[i]    = take((N,))

    sol_V = sol_V_re + 1j * sol_V_im
    sol_I = sol_I_re + 1j * sol_I_im
    sol_S = sol_V * np.conj(sol_I)

    traj = Trajectory(T, dt, {
        "voltage": n_buses, "current": n_buses, "power": n_buses,
        "delta": n_gens, "omega": n_gens, "Tm": n_gens, "Pc": n_gens,
    }, dtype=np.complex64)
    traj.w["voltage"] = sol_V
    traj.w["current"] = sol_I
    traj.w["power"]   = sol_S
    traj.w["delta"]   = sol_delta.astype(np.complex64)
    traj.w["omega"]   = sol_omega.astype(np.complex64)
    traj.w["Tm"]      = sol_Tm.astype(np.complex64)
    traj.w["Pc"]      = sol_Pc.astype(np.complex64)

    nlp_fns = {
        "x": x,
        "f": obj,
        "g": g,
        "lbx": lbx,
        "ubx": ubx,
        "lbg": lbg,
        "ubg": ubg,
    }
    return traj, sol, nlp_fns


def check_kkt(sol, nlp_fns, tol: float = 1e-5):
    """
    Verify KKT conditions at the IPOPT solution.

    Checks:
      1. Stationarity:        ||∇_x L||_inf < tol
      2. Primal feasibility:  max constraint violation < tol
      3. Dual feasibility:    inequality multipliers have correct sign
      4. Complementary slack: |μ * g(x*)| ≈ 0
    """
    x_sym = nlp_fns["x"]
    f_sym = nlp_fns["f"]
    g_sym = nlp_fns["g"]
    lbg   = nlp_fns["lbg"]
    ubg   = nlp_fns["ubg"]
    lbx   = nlp_fns["lbx"]
    ubx   = nlp_fns["ubx"]

    lam_g_sym = ca.MX.sym("lam_g", g_sym.size1())  # type: ignore[attr-defined]
    lam_x_sym = ca.MX.sym("lam_x", x_sym.size1())  # type: ignore[attr-defined]
    L = f_sym + ca.dot(lam_g_sym, g_sym) + ca.dot(lam_x_sym, x_sym)
    grad_L = ca.gradient(L, x_sym)
    grad_L_fn = ca.Function("grad_L", [x_sym, lam_g_sym, lam_x_sym], [grad_L])

    x_val    = sol["x"].full().flatten()
    lam_g    = sol["lam_g"].full().flatten()
    lam_x    = sol["lam_x"].full().flatten()
    g_val    = sol["g"].full().flatten()

    grad_val = grad_L_fn(x_val, lam_g, lam_x).full().flatten() # type: ignore
    stationarity = np.max(np.abs(grad_val))

    # Primal feasibility: constraints outside [lbg, ubg]
    viol = np.maximum(lbg - g_val, 0.0) + np.maximum(g_val - ubg, 0.0)
    # Variable bounds
    viol_x = np.maximum(lbx - x_val, 0.0) + np.maximum(x_val - ubx, 0.0)
    primal_feas = max(np.max(viol), np.max(viol_x))

    # Dual feasibility: for g >= 0 constraints (lbg=0, ubg=inf), lam_g should be <= 0
    # (IPOPT convention: lam_g is the multiplier for the constraint g - lbg >= 0)
    # Equality constraints (lbg == ubg) can have any sign.
    is_eq   = (ubg - lbg) < 1e-10
    is_ineq_lb = (~is_eq) & np.isfinite(lbg)   # lower-bounded inequality
    is_ineq_ub = (~is_eq) & np.isfinite(ubg)   # upper-bounded inequality
    dual_viol_lb = np.max(np.maximum( lam_g[is_ineq_lb], 0.0)) if is_ineq_lb.any() else 0.0
    dual_viol_ub = np.max(np.maximum(-lam_g[is_ineq_ub], 0.0)) if is_ineq_ub.any() else 0.0
    dual_feas = max(dual_viol_lb, dual_viol_ub)

    # Complementary slackness: only check the finite side of each constraint.
    # ubg = inf for one-sided lower inequalities, so skip those for the ub check
    # (0 * inf = nan otherwise).
    has_lb = np.isfinite(lbg)
    has_ub = np.isfinite(ubg)
    comp_slack_lb = np.max(np.abs(lam_g[has_lb] * (g_val[has_lb] - lbg[has_lb]))) if has_lb.any() else 0.0
    comp_slack_ub = np.max(np.abs(lam_g[has_ub] * (ubg[has_ub] - g_val[has_ub]))) if has_ub.any() else 0.0
    comp_slack = max(comp_slack_lb, comp_slack_ub)

    print("\n--- KKT Conditions ---")
    print(f"  Stationarity  ||∇L||_inf : {stationarity:.3e}  {'PASS' if stationarity < tol else 'FAIL'}")
    print(f"  Primal feas   max viol   : {primal_feas:.3e}  {'PASS' if primal_feas < tol else 'FAIL'}")
    print(f"  Dual feas     max viol   : {dual_feas:.3e}  {'PASS' if dual_feas < tol else 'FAIL'}")
    print(f"  Comp. slack   max |μg|   : {comp_slack:.3e}  {'PASS' if comp_slack < tol else 'FAIL'}")
    kkt_ok = stationarity < tol and primal_feas < tol and dual_feas < tol and comp_slack < tol
    print(f"  KKT overall: {'SATISFIED' if kkt_ok else 'VIOLATED'} (tol={tol:.0e})")
    return kkt_ok


def check_sosc(sol, nlp_fns, tol: float = 1e-6, active_tol: float = 1e-4):
    """
    Check Second-Order Sufficient Conditions (SOSC) at the IPOPT solution.

    Computes the reduced Hessian Z^T H Z restricted to the null space of the
    active constraint Jacobian. Reports the minimum eigenvalue — positive means
    the solution is a strict local minimum.

    Warning: expensive for large problems (~14k variables); uses dense null-space
    computation after extracting only the active constraint rows.
    """
    x_sym  = nlp_fns["x"]
    f_sym  = nlp_fns["f"]
    g_sym  = nlp_fns["g"]
    lbg    = nlp_fns["lbg"]
    ubg    = nlp_fns["ubg"]
    lbx    = nlp_fns["lbx"]
    ubx    = nlp_fns["ubx"]

    lam_g_sym = ca.MX.sym("lam_g", g_sym.size1())  # type: ignore[attr-defined]
    L = f_sym + ca.dot(lam_g_sym, g_sym)
    H_sym, _ = ca.hessian(L, x_sym)
    H_fn = ca.Function("H", [x_sym, lam_g_sym], [H_sym])
    J_fn = ca.Function("J", [x_sym], [ca.jacobian(g_sym, x_sym)])

    x_val   = sol["x"].full().flatten()
    lam_g   = sol["lam_g"].full().flatten()
    g_val   = sol["g"].full().flatten()

    H_sp = H_fn(x_val, lam_g)
    J_sp = J_fn(x_val)
    H_num = np.array(H_sp.full()) # type: ignore
    J_num = np.array(J_sp.full()) # type: ignore

    # Identify active constraints: equalities always active; inequalities active when at bound
    is_eq      = (ubg - lbg) < 1e-10
    active_lb  = (~is_eq) & (np.abs(g_val - lbg) < active_tol)
    active_ub  = (~is_eq) & (np.abs(g_val - ubg) < active_tol)
    active     = is_eq | active_lb | active_ub

    # Also include variable bounds (treated as additional rows in A_active)
    bound_lb = np.abs(x_val - lbx) < active_tol
    bound_ub = np.abs(x_val - ubx) < active_tol
    bound_active = bound_lb | bound_ub

    A_g = J_num[active, :]
    n_x = x_val.shape[0]
    A_bnd = np.eye(n_x)[bound_active, :]
    A_active = np.vstack([A_g, A_bnd]) if A_bnd.shape[0] > 0 else A_g

    print(f"\n--- SOSC Check ---")
    print(f"  Active constraints: {active.sum()} / {len(active)}  (+ {bound_active.sum()} bound rows)")

    if A_active.shape[0] >= n_x:
        print("  WARNING: active set spans full space — null space is trivial, SOSC trivially satisfied")
        return True

    Z = null_space(A_active)
    rH = Z.T @ H_num @ Z
    eigs = np.linalg.eigvalsh(rH)
    lam_min = eigs.min()
    print(f"  Reduced Hessian size: {rH.shape[0]}×{rH.shape[1]}")
    print(f"  Min eigenvalue: {lam_min:.6f}  {'PASS (local min)' if lam_min > -tol else 'FAIL (not local min)'}")
    return lam_min > -tol


def check_condition_number(sol, nlp_fns, log: logging.Logger | None = None):
    """
    Report the condition number of the constraint Jacobian and the Hessian of
    the Lagrangian at the IPOPT solution, via SVD (s_max / s_min over nonzero
    singular values). A large condition number indicates the NLP is poorly
    scaled/conditioned near the solution.

    Warning: expensive for large problems (dense SVD over a matrix that can be
    ~14k columns wide), same caveat as check_sosc.
    """
    x_sym = nlp_fns["x"]
    f_sym = nlp_fns["f"]
    g_sym = nlp_fns["g"]

    lam_g_sym = ca.MX.sym("lam_g", g_sym.size1())  # type: ignore[attr-defined]
    L = f_sym + ca.dot(lam_g_sym, g_sym)
    H_sym, _ = ca.hessian(L, x_sym)
    H_fn = ca.Function("H_cond", [x_sym, lam_g_sym], [H_sym])
    J_fn = ca.Function("J_cond", [x_sym], [ca.jacobian(g_sym, x_sym)])

    x_val  = sol["x"].full().flatten()
    lam_g  = sol["lam_g"].full().flatten()

    J_num = np.array(J_fn(x_val).full())       # type: ignore
    H_num = np.array(H_fn(x_val, lam_g).full())  # type: ignore

    def svd_cond(M, tol=1e-12):
        s = np.linalg.svd(M, compute_uv=False)
        s_nz = s[s > tol]
        s_min = s_nz.min() if s_nz.size else 0.0
        s_max = s.max()
        cond = s_max / s_min if s_min > 0 else np.inf
        return s_min, s_max, cond

    j_min, j_max, cond_J = svd_cond(J_num)
    h_min, h_max, cond_H = svd_cond(H_num)

    print("\n--- Condition Number ---")
    print(f"  Constraint Jacobian  shape={J_num.shape}  sigma_min={j_min:.3e}  sigma_max={j_max:.3e}  cond={cond_J:.3e}")
    print(f"  Lagrangian Hessian   shape={H_num.shape}  sigma_min={h_min:.3e}  sigma_max={h_max:.3e}  cond={cond_H:.3e}")

    if log is not None:
        log.info(f"Condition number - Jacobian: shape={J_num.shape} sigma_min={j_min:.3e} sigma_max={j_max:.3e} cond={cond_J:.3e}")
        log.info(f"Condition number - Hessian: shape={H_num.shape} sigma_min={h_min:.3e} sigma_max={h_max:.3e} cond={cond_H:.3e}")

    return cond_J, cond_H


def multi_start_verify(
    gens, loads, load_buses, sys_params, ic,
    n_starts: int = 5,
    noise_scale: float = 0.05,
    seed: int = 0,
    x0_override:  np.ndarray | None = None,
):
    """
    Solve the DA-OPF from multiple random starting points and compare primal
    solution vectors.

    n_starts perturbed warm starts are generated by adding scaled Gaussian noise
    to the steady-state warm start. All feasible solutions are compared; if the
    primal solution vectors agree (within a small norm threshold), global
    optimality confidence is high.
    """
    rng = np.random.default_rng(seed)

    # Solve once with the standard warm start to get the reference solution
    print("\n--- Multi-Start Global Optimality Check ---")
    print(f"  Running {n_starts} additional starts (noise_scale={noise_scale})...")

    diff_threshold = 1e-3
    max_diff = 0.0

    quiet_opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.tol": 1e-6,
        "ipopt.max_iter": 5000,
        "ipopt.acceptable_tol": 1e-5,
        "ipopt.acceptable_iter": 10,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.nlp_scaling_method": "gradient-based",
    }

    # Build the default warm start once to know its size, then perturb it
    _, ref_sol, ref_nlp = solve_daopf(
        gens=gens, loads=loads, load_buses=load_buses,
        sys_params=sys_params, ic=ic,
        opts_override=quiet_opts, x0_override=x0_override,
    )
    ref_x0 = ref_sol["x"].full().flatten()
    x_range = ref_nlp["ubx"] - ref_nlp["lbx"]
    x_range = np.where(np.isfinite(x_range), x_range, 2.0)

    for start_idx in range(n_starts):
        noise = rng.standard_normal(ref_x0.shape) * noise_scale * x_range
        x0_noisy = np.clip(ref_x0 + noise, ref_nlp["lbx"], ref_nlp["ubx"])
        _, sol_s, _ = solve_daopf(
            gens=gens, loads=loads, load_buses=load_buses,
            sys_params=sys_params, ic=ic,
            x0_override=x0_noisy,
            opts_override=quiet_opts,
        )
        x_s = sol_s["x"].full().flatten()
        diff = float(np.linalg.norm(x_s - ref_x0))
        max_diff = max(max_diff, diff)
        print(f"  Start {start_idx+1:2d}: ||x - x_ref|| = {diff:.6e}")

    print(f"\n  Max primal solution deviation: {max_diff:.6e}")
    print(f"  Threshold:                     {diff_threshold:.6e}")
    likely_global = max_diff < diff_threshold
    if likely_global:
        print("  Global optimality: LIKELY (all primal solutions match)")
    else:
        print("  Global optimality: NOT GLOBAL (primal solutions disagree)")
    return likely_global


def run_daopf_test(
    n_buses: int = 24,
    verify_kkt: bool = False,
    verify_sosc: bool = False,
    verify_global: bool = False,
    verify_condition: bool = False,
    load_admm_ics: bool = False
):
    assert n_buses % 2 == 0, "n_buses must be even"

    # Use SysParams for network topology and power targets
    sys_params = SysParams(n_buses)
    Ybus    = sys_params.Ybus
    S_gen0  = sys_params.S_gen0
    S_load0 = sys_params.S_load0
    n_gens  = sys_params.n_gens
    n_loads = sys_params.n_loads
    omega_s = sys_params.omega_s

    # Solve static OPF to get a power-flow-consistent operating point
    print("Solving static OPF for initial conditions...")
    opf_sol = solve_opf(sys_params)
    print(f"OPF status: {opf_sol['status']}")
    S_init = np.concatenate([np.full(n_gens, S_gen0), np.full(n_loads, S_load0)])
    ic = ic_from_opf(opf_sol, sys_params, S_init)

    print("Initial conditions from OPF:")
    print(f"Voltages (magnitude): {np.abs(ic['voltage'])}")
    print(f"Voltages (angle): {np.angle(ic['voltage'], deg=True)}")
    print(f"Currents (magnitude): {np.abs(ic['current'])}")
    print(f"Currents (angle): {np.angle(ic['current'], deg=True)}")
    print(f"Power injections: {ic['power']}")

    # Load disturbance at t=0.5s
    rng = np.random.default_rng(42)
    dP_loads    = rng.uniform(-0.2, 0.2, n_loads)
    P_load_post = np.real(S_load0) + dP_loads
    Q_load0     = np.imag(S_load0)

    # Generator EMF from OPF solution (consistent with Ybus @ V = I)
    E_complex = ic['voltage'][:n_gens] + 1j * sys_params.X_p * ic['current'][:n_gens]
    E_mags    = np.abs(E_complex)

    gen_costs = list(sys_params.gen_costs)
    P_min = sys_params.P_min
    P_max = sys_params.P_max
    V_max = sys_params.V_max

    gens = [
        Generator(E_mags[i], 1j*sys_params.X_p, sys_params.H, sys_params.D, sys_params.Tsv, sys_params.Rd, ic['delta'][i], ic['Tm'][i],
                  ic['voltage'][i], ic['current'][i], ic['power'][i],
                  Pc_min=P_min[i], Pc_max=P_max[i])
        for i in range(n_gens)
    ]
    loads = [
        ConstPowerLoad(
            lambda t, p0=np.real(S_load0), p1=float(P_load_post[j]), q=Q_load0:
                p0 + 1j*q if t < 0.5 else p1 + 1j*q
        )
        for j in range(n_loads)
    ]
    load_buses = list(range(n_gens, n_buses))

    log = logging.getLogger("daopf_benchmark")
    log.setLevel(logging.INFO)
    fh = logging.FileHandler("daopf_benchmark.log", mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(fh)

    x0_override = None
    if load_admm_ics and os.path.exists("admm_sols.pkl"):
        with open("admm_sols.pkl", "rb") as f:
            admm_sols = pickle.load(f)
        if n_buses in admm_sols:
            admm_sol = admm_sols[n_buses]
            print(f"Loading ADMM solution for {n_buses}-bus system as warm start.")
            # Construct x0_override from ADMM solution
            x0_V_re = np.real(admm_sol.w["voltage"]).flatten()
            x0_V_im = np.imag(admm_sol.w["voltage"]).flatten()
            x0_I_re = np.real(admm_sol.w["current"]).flatten()
            x0_I_im = np.imag(admm_sol.w["current"]).flatten()
            x0_parts = [x0_V_re, x0_V_im, x0_I_re, x0_I_im]
            for i in range(n_gens):
                x0_parts += [
                    np.real(admm_sol.w["delta"][i, :]),
                    np.real(admm_sol.w["omega"][i, :]),
                    np.real(admm_sol.w["Tm"][i, :]),
                    np.real(admm_sol.w["Pc"][i, :]),
                ]
            x0_override = np.concatenate(x0_parts)
        else:
            print(f"No ADMM solution found for {n_buses}-bus system. Starting from OPF ICs.")
    else:
        print("No ADMM solutions file found. Starting from OPF ICs.")
        


    t0 = time.perf_counter()
    sol, ipopt_sol, nlp_fns = solve_daopf(
        gens=gens,
        loads=loads,
        load_buses=load_buses,
        sys_params=sys_params,
        ic=ic,
        x0_override=x0_override
    )
    elapsed = time.perf_counter() - t0
    log.info("Number of Buses: " + str(n_buses))
    log.info(f"solve_daopf: {elapsed:.3f}s")
    print(f"DA-OPF solve time: {elapsed:.3f}s (logged to daopf_benchmark.log)")

    if verify_kkt:
        check_kkt(ipopt_sol, nlp_fns)

    if verify_sosc:
        check_sosc(ipopt_sol, nlp_fns)

    if verify_condition:
        check_condition_number(ipopt_sol, nlp_fns, log=log)

    if verify_global:
        multi_start_verify(gens, loads, load_buses, sys_params, ic, n_starts=20, noise_scale=0.05, x0_override=x0_override)

    t_vec = np.arange(sol.N) * sol.dt
    V_mag = np.abs(sol.w["voltage"])
    P     = np.real(sol.w["power"])
    Q     = np.imag(sol.w["power"])

    kcl_res = np.linalg.norm(Ybus @ sol.w["voltage"] - sol.w["current"], axis=0)
    load_res_per_bus = [
        np.abs(
            sol.w["voltage"][n_gens + j, :] * np.conj(sol.w["current"][n_gens + j, :])
            - np.array([loads[j].S(k * sol.dt) for k in range(sol.N)])
        )
        for j in range(n_loads)
    ]
    load_res = np.max(load_res_per_bus, axis=0)

    print(f"Max KCL residual:        {kcl_res.max():.3e}")
    print(f"Max load power residual: {load_res.max():.3e}")

    # Figure 1: Generator dynamics
    fig1, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle(f"DA-OPF (Centralized) — Generator Dynamics ({n_buses}-bus)")

    for i in range(n_gens):
        axs[0, 0].plot(t_vec, np.real(sol.w["omega"][i, :]), label=f"Gen {i+1}")
    axs[0, 0].axhline(omega_s, color="k", linestyle="--", linewidth=0.8, label="ωs")
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
        ax.axvline(0.5, color="r", linestyle=":", linewidth=0.8)
        ax.grid(True)
    plt.tight_layout()

    # Figure 2: Power
    fig2, axs2 = plt.subplots(2, 1, figsize=(12, 8))
    fig2.suptitle(f"DA-OPF (Centralized) — Bus Power ({n_buses}-bus)")

    for bus in range(n_buses):
        lbl = f"Gen {bus+1}" if bus < n_gens else f"Load {bus-n_gens+1}"
        axs2[0].plot(t_vec, P[bus, :], label=lbl)
    axs2[0].axvline(0.5, color="r", linestyle=":", linewidth=0.8, label="disturbance")
    axs2[0].set_ylabel("P (pu)")
    axs2[0].set_title("Real Power")
    axs2[0].legend(fontsize=6)
    axs2[0].grid(True)

    for bus in range(n_buses):
        lbl = f"Gen {bus+1}" if bus < n_gens else f"Load {bus-n_gens+1}"
        axs2[1].plot(t_vec, Q[bus, :], label=lbl)
    axs2[1].axvline(0.5, color="r", linestyle=":", linewidth=0.8)
    axs2[1].set_ylabel("Q (pu)")
    axs2[1].set_title("Reactive Power")
    axs2[1].set_xlabel("Time (s)")
    axs2[1].legend(fontsize=6)
    axs2[1].grid(True)
    plt.tight_layout()

    # Figure 3: Feasibility residuals
    fig3, axs3 = plt.subplots(2, 1, figsize=(10, 6))
    fig3.suptitle("DA-OPF (Centralized) — Constraint Residuals")

    axs3[0].semilogy(t_vec, kcl_res + 1e-16)
    axs3[0].set_xlabel("Time (s)")
    axs3[0].set_ylabel("‖Ybus·V − I‖")
    axs3[0].set_title("KCL Residual over Time")
    axs3[0].grid(True)

    axs3[1].semilogy(t_vec, load_res + 1e-16)
    axs3[1].set_xlabel("Time (s)")
    axs3[1].set_ylabel("|V·I* - S_load| (max over loads)")
    axs3[1].set_title("Load Power Constraint Residual")
    axs3[1].grid(True)

    plt.tight_layout()
    
    # Figure 4: Pc
    fig4, axs4 = plt.subplots(2, 1, figsize=(10, 6))
    fig4.suptitle("DA-OPF (Centralized) — Pc")
    for i in range(n_gens):
        axs4[0].plot(t_vec, np.real(sol.w["Pc"][i, :]), label=f"Gen {i+1}")
    axs4[0].set_ylabel("Pc (pu)")
    axs4[0].set_title("Governor Power Setpoint")
    axs4[0].legend(fontsize=7)
    axs4[0].grid(True)

    for i in range(n_gens):
        axs4[1].plot(t_vec, np.real(sol.w["Pc"][i, :]) - np.real(sol.w["Pc"][i, 0]), label=f"Gen {i+1}")
    axs4[1].set_ylabel("ΔPc (pu)")
    axs4[1].set_title("Change in Power Setpoint")
    axs4[1].set_xlabel("Time (s)")
    axs4[1].legend(fontsize=7)
    axs4[1].grid(True)
    plt.tight_layout()
    

    plt.show()

    return elapsed, sol


if __name__ == "__main__":
    import pickle

    times = {}
    if os.path.exists("daopf_times.pkl"):
        with open("daopf_times.pkl", "rb") as f:
            times = pickle.load(f)

    busses = 4
    timing_results, sol = run_daopf_test(
        n_buses=busses, verify_kkt=True,
        verify_sosc=False,
        verify_global=False,
        verify_condition=True,
        load_admm_ics=False
    )
    times[busses] = timing_results

    # Save times dict to file
    import pickle
    with open("daopf_times.pkl", "wb") as f:
        pickle.dump(times, f)
    
    # Save Trajectory solution to file
    sols = {}
    if os.path.exists("daopf_sols.pkl"):
        with open("daopf_sols.pkl", "rb") as f:
            sols = pickle.load(f)
    sols[busses] = sol
    # Save sols dict to file
    with open("daopf_sols.pkl", "wb") as f:
        pickle.dump(sols, f)