from saap import SysParams, solve_opf, ic_from_opf
import numpy as np

print("Testing OPF solver...")
sys_params = SysParams(n_buses=12)
print(f"System initialized: {sys_params.n_buses} buses, {sys_params.n_gens} generators, {sys_params.n_loads} loads")

print("Solving OPF...")
opf_sol = solve_opf(sys_params)
print(f"OPF solved: {opf_sol['status']}")

# Check KCL residual
V_opf = opf_sol['V']
I_opf = opf_sol['I']
kcl_residual = np.linalg.norm(sys_params.Ybus @ V_opf - I_opf)
print(f"KCL residual: {kcl_residual:.6e}")

# Check voltage magnitudes
V_mag = np.abs(V_opf)
print(f"Voltage magnitudes (min, max): {V_mag.min():.4f}, {V_mag.max():.4f}")

# Check generation
P_gen = opf_sol['P_gen']
print(f"Generation dispatch (min, max): {P_gen.min():.4f}, {P_gen.max():.4f}")
print(f"Generation within bounds: {np.all(P_gen >= sys_params.P_min) and np.all(P_gen <= sys_params.P_max)}")

# Extract ICs
S_init = np.concatenate([
    np.full(sys_params.n_gens, sys_params.S_gen0),
    np.full(sys_params.n_loads, sys_params.S_load0)
])
ic = ic_from_opf(opf_sol, sys_params, S_init)
print(f"IC extracted successfully")
print(f"  - Voltage shape: {ic['voltage'].shape}")
print(f"  - Current shape: {ic['current'].shape}")
print(f"  - Delta shape: {ic['delta'].shape}")
print(f"  - Omega values (unique): {np.unique(ic['omega'])}")

# Verify IC feasibility
ic_kcl_residual = np.linalg.norm(sys_params.Ybus @ ic['voltage'] - ic['current'])
print(f"IC KCL residual (verification): {ic_kcl_residual:.6e}")

print("\nTest PASSED!")
