# Plan: Diagonal Preconditioning for ADMM

## Context

The ADMM solver in [saap.py](saap.py) splits the OPF problem into a CVXPY QP step (`Objective.prox`, the x-update) and a per-bus NLP projection step (`Generator.prox`/`ConstPowerLoad.prox`, the z-update). All consensus variables — voltage, current, power, `Pc`, `omega`, `delta`, `Tm` — are stacked into one `Trajectory` and compared with a single, unweighted Euclidean norm (`Trajectory.norm`, [Trajectory.py:145-148](Trajectory.py#L145-L148)) everywhere ADMM needs a distance: the QP penalty ([Objective.py:44-46](Objective.py#L44-L46)), the NLP projection objective in `Generator.prox` (`sum_sqr(...)` over all 10 stacked variables), the primal/dual residuals, and the dual-variable update in [saap.py:79-88](saap.py#L79-L88).

These variables live on wildly different physical scales (`omega` ~377 rad/s vs `delta` ~0.1 rad vs `V`,`I`,`P`,`Q`,`Pc` ~O(1) pu). A single scalar `rho` cannot simultaneously be "the right" penalty weight for all of them — it is too stiff for some blocks and too loose for others, which is the classic source of the slow/plateauing convergence seen in `admm_benchmark.log` (14 iterations historically vs 400-850 in recent runs). The fix is to replace the single scalar `rho` with a **diagonal preconditioner**: a fixed per-variable weight `D_v` such that the effective penalty becomes `(rho/2) * D_v^2 * ||x_v - z_v||^2` for each variable block `v`. This is standard generalized ADMM (block-diagonal rho) and requires no change to the underlying physical constraints (Ybus, voltage/power bounds, generator DAEs) — only to the distance metric ADMM uses to drive consensus.

Along the way, a real bug was found: `Objective.py:34` hardcodes `self.rho = 2.0` as a Python float baked into the CVXPY problem at construction time, and the line that would wire the passed-in `rho` argument into a `cp.Parameter` is commented out ([Objective.py:73](Objective.py#L73)). This means `rho_heuristic`'s adaptive updates in `saap.py` never actually reach the QP's penalty term — only the dual-variable rescaling (line 77) sees them. This must be fixed for any rho/preconditioning scheme to actually work.

## Approach

### 1. Fix the rho propagation bug in `Objective.py`
- Change `self.rho = 2.0` to `self.rho = cp.Parameter(nonneg=True)` (uncomment/restore the parameter declaration already stubbed at line 33).
- In `prox`, set `self.rho.value = rho` (restore line 73) before `self.problem.solve()`.
- Verify `cp.Problem` still compiles with `rho` as a `Parameter` (CVXPY supports this natively; no DPP issues expected since `rho` only scales the penalty terms).

### 2. Add per-variable preconditioning weights
- Add a method to `SysParams` (or a small new module, e.g. `Precondition.py`) that returns a dict of scale weights keyed by variable name, e.g.:
  ```python
  def precondition_weights(self) -> dict[str, float]:
      return {
          "voltage": 1.0,
          "current": 1.0,
          "power": 1.0,
          "Pc": 1.0,
          "omega": 1.0 / self.omega_band,   # omega only varies by +/- omega_band, not by omega_s
          "delta": 1.0,
          "Tm": 1.0,
      }
  ```
  The key insight (confirmed while reading `SysParams.py`): `omega`'s large absolute value (~377 rad/s) is a constant offset shared by both `x` and `z`, so it mostly cancels in `x - z`. The real scale mismatch is in the *range of variation* each variable is allowed (`omega_band` ≈ 0.08 vs `delta`, `V`, `P` ranges ≈ O(1)). Weights should be derived from each variable's natural operating range (use existing `P_min`/`P_max`, `V_max`, `omega_band` from `SysParams`), not raw magnitude — confirm this against the actual residual plateau before committing to fixed numbers (see Verification below).
- These weights are static (computed once from `SysParams`), not adapted iteration-to-iteration — keep the existing `rho_heuristic` scalar adaptation as the global scale, multiplied by the fixed per-variable weight.

### 3. Thread weighted penalties through the x-update (`Objective.py`)
- Replace the single `self.penalty` (line 44) with per-block penalties, each scaled by its weight `D_v^2`:
  ```python
  self.penalty = self.rho/2 * D["network"] * cp.sum_squares(self.x - self.w)  # or split V/I/S individually if weights differ
  self.Pc_penalty = self.rho/2 * D["Pc"] * cp.sum_squares(self.Pc_var - self.Pcw)
  self.omega_penalty = self.rho/2 * D["omega"] * cp.sum_squares(self.omega - self.omegaw)
  ```
  Pass the weights dict into `Objective.__init__`.

### 4. Thread weighted penalties through the z-update (`Generator.py`)
- In `Generator.prox`, the projection objective at line ~172 (`sum_sqr(all vars - measurement)`) currently weights `delta`, `omega`, `Tm`, `V_re/im`, `I_re/im`, `P`, `Q`, `Pc` equally. Replace with a weighted sum: `sum_sqr(weight_v * (var - var_traj))` per variable block, using the same weights dict from step 2.
- `ConstPowerLoad.prox` uses a closed-form analytic projection (not a generic sum-of-squares NLP) — re-deriving it for weighted V/I norms is out of scope for this pass. Since `voltage` and `current` get equal weight 1.0 in step 2, leave `ConstPowerLoad.prox` unchanged; note this as a known limitation.

### 5. Update residual and dual computation in `saap.py`
- `Trajectory.norm()` is unweighted; add a `weighted_norm(weights: dict)` method to `Trajectory` (or a free function) that scales each block by `sqrt(weight)` before taking the norm.
- Update the primal/dual residual lines ([saap.py:87-88](saap.py#L87-L88)) and `rho_heuristic`'s norm calls to use the weighted norm, so the adaptive-rho heuristic sees a metric consistent with the actual penalty being applied.
- The dual update (`us[-1] + (xs[-1] - zs[-1])`, line 83) should remain an unweighted accumulation of the *unweighted* primal gap — only the penalty term and the convergence-residual metric need weighting (this matches how generalized ADMM with diagonal rho is normally written: `u += D*(x - z)` if you want strict equivalence with scaled variables, but the simpler and more common implementation keeps `u` unweighted and just applies `D` inside the prox objectives and residual norms). Decide finally based on which keeps the rho-rescaling-on-change logic in `saap.py:76-77` correct; default to weighting only the objectives/residuals to avoid touching the rescale-on-rho-change formula.

### 6. Wire weights through `_setup_admm_problem` / `admm_test`
- Compute `weights = sys_params.precondition_weights()` once in `_setup_admm_problem`, pass to `Objective(...)` and to each `Generator(...)` (or pass through `BusBehaviours`/`BusBehavioursSerial`/`BusBehavioursParallel` constructors down to `gen.prox`).

## Verification
- Before changing weights, first land just the `Objective.py` rho-propagation bugfix (step 1) alone and rerun `admm_test(n_buses=4, ...)` to see if simply letting rho adapt correctly already reduces iteration count — this isolates the bug fix's effect from the preconditioning effect.
- Then add the diagonal weights and rerun the same benchmark (`python saap.py`, which calls `admm_test` and appends to `admm_benchmark.log`). Compare iteration counts and final residuals before/after via the existing `admm_benchmark.log` entries.
- Sanity-check that the final converged trajectory's physical quantities (KCL residual, load power residual, generator dynamics) are unchanged/still satisfy constraints — preconditioning only changes the convergence path, not the optimum, so `compute_cost` and the existing `fig3`/`fig4` residual plots in `admm_test` should look the same or better, not different.
- Try a couple of candidate weight sets (e.g. weight by inverse of `P_max`/`V_max`/`omega_band` vs. weight by inverse of observed residual magnitude per block from a baseline run) and pick whichever gives the largest iteration-count reduction on the 4-bus and 24-bus cases.
