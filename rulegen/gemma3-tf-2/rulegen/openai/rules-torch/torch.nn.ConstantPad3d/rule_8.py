import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# avoid overflow and excessive allocation with uniform int padding by bounding output sizes (Rule 8)

rule_8 = lambda s, v, n=False: (
    s.add(Not(Or((And(And(And(And(And(v["arg2_ndim"] == 5, Select(v["arg2_shape"], 0) <= 4096), Select(v["arg2_shape"], 1) <= 8192), Select(v["arg2_shape"], 2) + 2 * v["arg1_value"] <= 1048576), Select(v["arg2_shape"], 3) + 2 * v["arg1_value"] <= 1048576), Select(v["arg2_shape"], 4) + 2 * v["arg1_value"] <= 1048576)), (And(And(And(And(v["arg2_ndim"] == 4, Select(v["arg2_shape"], 0) <= 8192), Select(v["arg2_shape"], 1) + 2 * v["arg1_value"] <= 1048576), Select(v["arg2_shape"], 2) + 2 * v["arg1_value"] <= 1048576), Select(v["arg2_shape"], 3) + 2 * v["arg1_value"] <= 1048576)))) if n else
          Or((And(And(And(And(And(v["arg2_ndim"] == 5, Select(v["arg2_shape"], 0) <= 4096), Select(v["arg2_shape"], 1) <= 8192), Select(v["arg2_shape"], 2) + 2 * v["arg1_value"] <= 1048576), Select(v["arg2_shape"], 3) + 2 * v["arg1_value"] <= 1048576), Select(v["arg2_shape"], 4) + 2 * v["arg1_value"] <= 1048576)), (And(And(And(And(v["arg2_ndim"] == 4, Select(v["arg2_shape"], 0) <= 8192), Select(v["arg2_shape"], 1) + 2 * v["arg1_value"] <= 1048576), Select(v["arg2_shape"], 2) + 2 * v["arg1_value"] <= 1048576), Select(v["arg2_shape"], 3) + 2 * v["arg1_value"] <= 1048576))))
)

def rule_8_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 8
        rule_8(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_8(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']}, neg)
