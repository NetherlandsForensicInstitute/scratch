"""Compare custom fminsearch vs scipy Nelder-Mead."""
import sys
sys.path.insert(0, 'packages/scratch-core/src')

import numpy as np
from scipy.optimize import minimize
from conversion.profile_correlator.alignment import (
    _apply_lowpass_filter_1d, _alignment_objective,
    _fminsearchbnd, _fminsearchbnd_transform_to_unconstrained,
    _fminsearchbnd_transform_to_bounded, _matlab_fminsearch,
)

base = 'packages/scratch-core/tests/resources/profile_correlator/edge_over_threshold'
ref_data = np.load(f'{base}/input_profile_ref.npy').ravel().astype(np.float64)
comp_data = np.load(f'{base}/input_profile_comp.npy').ravel().astype(np.float64)
pixel_size = 3.5e-6

p1 = ref_data[:460]
p2 = comp_data.copy()

p1_lp = _apply_lowpass_filter_1d(p1, 500e-6, pixel_size)
p2_lp = _apply_lowpass_filter_1d(p2, 500e-6, pixel_size)
p1_sub = p1_lp[::15]
p2_sub = p2_lp[::15]

lb = np.array([-33.0, -500.0])
ub = np.array([33.0, 500.0])
x0 = np.array([0.0, 0.0])

# Method 1: Custom fminsearchbnd
x1 = _fminsearchbnd(
    _alignment_objective, x0, lb, ub,
    tol_x=1e-6, tol_fun=1e-6, max_iter=400, max_fun_evals=400,
    args=(p1_sub, p2_sub),
)
print(f"Custom fminsearchbnd: t={x1[0]:.6f}, s_enc={x1[1]:.6f}")
print(f"  Bounded: trans={x1[0]:.4f}, scale={x1[1]/10000+1:.6f}")
obj1 = _alignment_objective(x1, p1_sub, p2_sub)
print(f"  Objective: {obj1:.6f} (corr={-obj1:.6f})")

# Method 2: scipy minimize with Nelder-Mead in transformed space
x0u = _fminsearchbnd_transform_to_unconstrained(x0, lb, ub)
print(f"\nTransformed x0: {x0u}")

def wrapped_fun(xu):
    x = _fminsearchbnd_transform_to_bounded(xu, lb, ub)
    return _alignment_objective(x, p1_sub, p2_sub)

result = minimize(wrapped_fun, x0u, method='Nelder-Mead',
                  options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 400, 'maxfev': 400})
x2 = _fminsearchbnd_transform_to_bounded(result.x, lb, ub)
print(f"\nScipy Nelder-Mead: t={x2[0]:.6f}, s_enc={x2[1]:.6f}")
print(f"  Bounded: trans={x2[0]:.4f}, scale={x2[1]/10000+1:.6f}")
obj2 = _alignment_objective(x2, p1_sub, p2_sub)
print(f"  Objective: {obj2:.6f} (corr={-obj2:.6f})")
print(f"  Converged: {result.success}, nfev={result.nfev}, nit={result.nit}")

# Method 3: Direct (non-wrapped) optimization
print(f"\n--- Checking initial simplex evaluations ---")
x0u = _fminsearchbnd_transform_to_unconstrained(x0, lb, ub)
print(f"x0u = {x0u}")
# Simplex vertices
usual_delta = 0.05
zero_term_delta = 0.00025
v0 = x0u.copy()
v1 = x0u.copy(); v1[0] *= (1 + usual_delta)
v2 = x0u.copy(); v2[1] *= (1 + usual_delta)
print(f"Simplex vertices (unconstrained):")
print(f"  v0 = {v0}")
print(f"  v1 = {v1}")
print(f"  v2 = {v2}")

# Transform to bounded
b0 = _fminsearchbnd_transform_to_bounded(v0, lb, ub)
b1 = _fminsearchbnd_transform_to_bounded(v1, lb, ub)
b2 = _fminsearchbnd_transform_to_bounded(v2, lb, ub)
print(f"Simplex vertices (bounded):")
print(f"  b0 = {b0}")
print(f"  b1 = {b1}")
print(f"  b2 = {b2}")

# Evaluate objectives
f0 = wrapped_fun(v0)
f1 = wrapped_fun(v1)
f2 = wrapped_fun(v2)
print(f"Simplex function values:")
print(f"  f(v0) = {f0:.10f}")
print(f"  f(v1) = {f1:.10f}")
print(f"  f(v2) = {f2:.10f}")
print(f"  Sort order: {np.argsort([f0, f1, f2])}")
