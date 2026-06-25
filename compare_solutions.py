"""
Cross-evaluation diagnostic between the monolithic DA-OPF (daopf.solve_daopf)
and the ADMM solver (saap.admm).

Runs both solvers on the same case and initial trajectory, then:
  1. Evaluates the ADMM solution against the daopf constraint vector,
     reporting max violation per constraint group (KCL, gen dynamics,
     algebraic, load power, P limits, V limits, variable bounds).
  2. Checks whether each solution is a fixed point of the bus-behaviour
     projection (i.e. membership in Bbus).
  3. Compares generation cost  sum_i c_i * Re(V_i conj(I_i))^2  evaluated
     identically on all trajectories.
  4. Reports min |V| of each solution to settle whether the |V| >= 0.8
     bound (present only in daopf) is ever active.

Usage: python compare_solutions.py [label]
The label is appended to the saved overlay plot filenames.
"""
import sys
import time

import numpy as np
import casadi as ca
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from SysParams import SysParams
from Trajectory import Trajectory
from Generator import Generator
from ConstPowerLoad import ConstPowerLoad
from BusBehaviours import BusBehaviours
from Objective import Objective
from OPF import solve_opf, ic_from_opf
from daopf import solve_daopf
from saap import admm, rho_heuristic

N_BUSES = 4
DTYPE = np.complex128
ADMM_MAX_ITER = 2000
ADMM_THRESHOLD = 1e-3


def build_case(n_buses=N_BUSES):
    sys_params = SysParams(n_buses)
    n_gens, n_loads = sys_params.n_gens, sys_params.n_loads

    opf_sol = solve_opf(sys_params)
    opf_post = solve_opf(sys_params, post=True)
    S_init = np.concatenate([np.full(n_gens, sys_params.S_gen0),
                             np.full(n_loads, sys_params.S_load0)])
    ic = ic_from_opf(opf_sol, sys_params, S_init)
    ic_post = ic_from_opf(opf_post, sys_params, S_init)

    E_complex = ic['voltage'][:n_gens] + 1j * sys_params.X_p * ic['current'][:n_gens]
    E_mags = np.abs(E_complex)
    gens = [
        Generator(E_mags[i], 1j*sys_params.X_p, sys_params.H, sys_params.D,
                  sys_params.Tsv, sys_params.Rd, ic['delta'][i], ic['Tm'][i],
                  ic['voltage'][i], ic['current'][i], ic['power'][i],
                  Pc_min=sys_params.P_min[i], Pc_max=sys_params.P_max[i])
        for i in range(n_gens)
    ]
    loads = [ConstPowerLoad(sys_params.get_load_power(j)) for j in range(n_loads)]
    load_buses = list(range(n_gens, n_buses))
    return sys_params, gens, loads, load_buses, ic, ic_post


def make_initial_traj(sys_params, ic, ic_post):
    """Same piecewise pre/post-disturbance initialization as saap.admm_test."""
    n_buses, n_gens = sys_params.n_buses, sys_params.n_gens
    t = Trajectory(sys_params.T, sys_params.dt, {
        "voltage": n_buses, "current": n_buses, "power": n_buses,
        "delta": n_gens, "omega": n_gens, "Tm": n_gens, "Pc": n_gens,
    }, dtype=DTYPE)
    t.set_constant(["voltage"], list(ic['voltage']))
    t.set_constant(["current"], list(ic['current']))
    t.set_constant(["delta"],   list(ic['delta']))
    t.set_constant(["omega"],   list(ic['omega']))
    t.set_constant(["Tm"],      list(ic['Tm']))
    t.set_constant(["power"],   list(ic['power']))
    t.set_constant(["Pc"],      list(ic['Pc']))
    for name in ["voltage", "current", "power", "delta", "omega", "Tm", "Pc"]:
        t.w[name][:, sys_params.N_step:] = ic_post[name].reshape(-1, 1)
    return t


def pack_daopf_x(traj, sys_params):
    """Pack a Trajectory into the daopf decision-vector ordering (daopf.py x_parts)."""
    n_gens = sys_params.n_gens
    V, I = traj.w["voltage"], traj.w["current"]
    parts = [np.real(V).flatten(order='F'), np.imag(V).flatten(order='F'),
             np.real(I).flatten(order='F'), np.imag(I).flatten(order='F')]
    for i in range(n_gens):
        parts += [np.real(traj.w["delta"][i]), np.real(traj.w["omega"][i]),
                  np.real(traj.w["Tm"][i]),    np.real(traj.w["Pc"][i])]
    return np.concatenate(parts)


def daopf_constraint_groups(sys_params):
    """(name, size) for each block of g, in daopf assembly order."""
    n_buses, n_gens, n_loads, N = (sys_params.n_buses, sys_params.n_gens,
                                   sys_params.n_loads, sys_params.N)
    groups = [("KCL (Ybus V = I)", 2 * n_buses * N)]
    for i in range(n_gens):
        groups += [
            (f"gen{i} initial conditions", 3),
            (f"gen{i} terminal (omega=ws)", 1),
            (f"gen{i} dynamics (trapezoidal)", 3 * (N - 1)),
            (f"gen{i} algebraic (E-Zd*I-V)", 2 * N),
        ]
    for j in range(n_loads):
        groups.append((f"load{j} power (V conj(I) = S)", 2 * N))
    for i in range(n_gens):
        groups.append((f"gen{i} P limits", 2 * N))
    groups.append(("V upper bound", n_buses * N))
    groups.append(("V lower bound (0.8)", n_buses * N))
    return groups


def eval_in_daopf(traj, nlp_fns, sys_params, label):
    x_val = pack_daopf_x(traj, sys_params)
    g_fn = ca.Function("g_eval", [nlp_fns["x"]], [nlp_fns["g"]])
    g_val = np.array(g_fn(x_val)).flatten()
    viol = np.maximum(nlp_fns["lbg"] - g_val, 0) + np.maximum(g_val - nlp_fns["ubg"], 0)

    print(f"\n--- {label}: violation of daopf constraints (max per group) ---")
    offset = 0
    worst = 0.0
    for name, size in daopf_constraint_groups(sys_params):
        v = viol[offset: offset + size].max() if size > 0 else 0.0
        worst = max(worst, v)
        print(f"  {name:38s}: {v:.3e}")
        offset += size
    assert offset == len(viol), f"group sizes {offset} != len(g) {len(viol)}"

    viol_x = (np.maximum(nlp_fns["lbx"] - x_val, 0)
              + np.maximum(x_val - nlp_fns["ubx"], 0)).max()
    print(f"  {'variable bounds (omega, Pc)':38s}: {viol_x:.3e}")
    print(f"  WORST overall: {max(worst, viol_x):.3e}")
    return max(worst, viol_x)


def gen_cost(traj, sys_params):
    n_gens = sys_params.n_gens
    P = np.real(traj.w["voltage"][:n_gens] * np.conj(traj.w["current"][:n_gens]))
    return float(np.sum(np.asarray(sys_params.gen_costs).reshape(-1, 1) * P**2))


def bbus_fixed_point_distance(traj, Bi, label):
    proj = Bi.prox(traj)
    diff = proj - traj
    print(f"\n--- {label}: distance to Bbus, ||prox(w) - w|| ---")
    total = diff.norm()
    for name in traj.vars:
        print(f"  {name:10s}: {np.linalg.norm(diff.w[name]):.3e}")
    print(f"  total     : {total:.3e}")
    return total


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "run"

    sys_params, gens, loads, load_buses, ic, ic_post = build_case()
    init_traj = make_initial_traj(sys_params, ic, ic_post)

    # ---------------- daopf, default warm start ----------------
    print("\n================ daopf (default warm start) ================")
    t0 = time.perf_counter()
    traj_d, sol_d, nlp_fns = solve_daopf(gens, loads, load_buses, sys_params, ic)
    print(f"daopf default solve: {time.perf_counter()-t0:.1f}s, obj = {float(sol_d['f']):.6f}")

    # ---------------- daopf, warm-started from the ADMM initial trajectory ----------------
    print("\n================ daopf (ADMM-aligned warm start) ================")
    t0 = time.perf_counter()
    traj_d2, sol_d2, _ = solve_daopf(gens, loads, load_buses, sys_params, ic,
                                     x0_override=pack_daopf_x(init_traj, sys_params))
    print(f"daopf aligned solve: {time.perf_counter()-t0:.1f}s, obj = {float(sol_d2['f']):.6f}")

    # ---------------- ADMM ----------------
    print("\n================ ADMM ================")
    obj = Objective(sys_params.Ybus, sys_params.gen_costs, sys_params.P_min,
                    sys_params.P_max, sys_params.V_max, init_traj, omega_s=sys_params.omega_s)
    Bi = BusBehaviours(gens, loads)
    iters = {"n": 0, "r": np.nan, "s": np.nan}

    def cb(iteration, x, z, u, r, s):
        iters["n"], iters["r"], iters["s"] = iteration + 1, r, s
        if (iteration + 1) % 10 == 0 or iteration == 0:
            print(f"  ADMM iter {iteration+1}: r = {r:.4e}, s = {s:.4e}")

    t0 = time.perf_counter()
    traj_a = admm(obj, Bi, init_traj, rho=rho_heuristic, # type: ignore
                  threshold=ADMM_THRESHOLD, max_iterations=ADMM_MAX_ITER, callback=cb)
    print(f"ADMM: {time.perf_counter()-t0:.1f}s, {iters['n']} iterations, "
          f"final r = {iters['r']:.4e}, s = {iters['s']:.4e}")

    # ---------------- Cross-evaluation ----------------
    eval_in_daopf(traj_d,  nlp_fns, sys_params, "daopf default (sanity)")
    eval_in_daopf(traj_d2, nlp_fns, sys_params, "daopf aligned (sanity)")
    eval_in_daopf(traj_a,  nlp_fns, sys_params, "ADMM solution")

    bbus_fixed_point_distance(traj_d2, Bi, "daopf aligned solution")
    bbus_fixed_point_distance(traj_a,  Bi, "ADMM solution")

    print("\n--- Generation cost  sum_i c_i * Re(V_i conj I_i)^2 ---")
    print(f"  daopf default : {gen_cost(traj_d,  sys_params):.6f}")
    print(f"  daopf aligned : {gen_cost(traj_d2, sys_params):.6f}")
    print(f"  ADMM          : {gen_cost(traj_a,  sys_params):.6f}")

    print("\n--- min |V| over all buses/timesteps (daopf enforces >= 0.8) ---")
    for name, tr in [("daopf default", traj_d), ("daopf aligned", traj_d2), ("ADMM", traj_a)]:
        print(f"  {name:14s}: {np.abs(tr.w['voltage']).min():.4f}")

    # ---------------- Overlay plots ----------------
    n_gens, n_buses, omega_s = sys_params.n_gens, sys_params.n_buses, sys_params.omega_s
    t_vec = np.arange(traj_a.N) * traj_a.dt
    styles = [("daopf default", traj_d, "-"), ("daopf aligned", traj_d2, "--"), ("ADMM", traj_a, ":")]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"daopf vs ADMM overlay ({N_BUSES}-bus) [{label}]")
    for sname, tr, ls in styles:
        for i in range(n_gens):
            axs[0, 0].plot(t_vec, np.real(tr.w["omega"][i]), ls, label=f"{sname} G{i+1}")
            axs[0, 1].plot(t_vec, np.degrees(np.real(tr.w["delta"][i])), ls, label=f"{sname} G{i+1}")
            axs[1, 0].plot(t_vec, np.real(tr.w["Tm"][i]), ls, label=f"{sname} G{i+1}")
        for b in range(n_buses):
            axs[1, 1].plot(t_vec, np.abs(tr.w["voltage"][b]), ls, label=f"{sname} B{b+1}")
    axs[0, 0].axhline(omega_s, color="k", linewidth=0.8)
    axs[0, 0].set_title("Rotor speed ω")
    axs[0, 1].set_title("Rotor angle δ (deg)")
    axs[1, 0].set_title("Mechanical torque Tm")
    axs[1, 1].set_title("|V|")
    for ax in axs.flat:
        ax.set_xlabel("Time (s)")
        ax.grid(True)
        ax.legend(fontsize=6)
    plt.tight_layout()
    fig.savefig(f"compare_overlay_dynamics_{label}.png", dpi=130)

    fig2, axs2 = plt.subplots(2, 1, figsize=(12, 8))
    fig2.suptitle(f"daopf vs ADMM bus power overlay ({N_BUSES}-bus) [{label}]")
    for sname, tr, ls in styles:
        S = tr.w["voltage"] * np.conj(tr.w["current"])
        for b in range(n_buses):
            axs2[0].plot(t_vec, np.real(S[b]), ls, label=f"{sname} B{b+1}")
            axs2[1].plot(t_vec, np.imag(S[b]), ls, label=f"{sname} B{b+1}")
    axs2[0].set_title("Re(V conj I)")
    axs2[1].set_title("Im(V conj I)")
    for ax in axs2:
        ax.set_xlabel("Time (s)")
        ax.grid(True)
        ax.legend(fontsize=6)
    plt.tight_layout()
    fig2.savefig(f"compare_overlay_power_{label}.png", dpi=130)

    print(f"\nSaved overlay plots: compare_overlay_dynamics_{label}.png, "
          f"compare_overlay_power_{label}.png")


if __name__ == "__main__":
    main()
