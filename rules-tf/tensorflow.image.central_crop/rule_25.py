import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Image tensor must have valid dtype, dimensions, and shape based on central_fraction (Rule 25)

rule_25 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg2_value"] > 0.0, v["arg2_value"] <= 1.0), (Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 6))), (Or((And(And(And(And(And(And(v["arg1_ndim"] == 3, Select(v["arg1_shape"], 0) > 0), Select(v["arg1_shape"], 1) > 0), 0 < Select(v["arg1_shape"], 0) * v["arg2_value"]), Select(v["arg1_shape"], 0) * v["arg2_value"] <= Select(v["arg1_shape"], 0)), 0 < Select(v["arg1_shape"], 1) * v["arg2_value"]), Select(v["arg1_shape"], 1) * v["arg2_value"] <= Select(v["arg1_shape"], 1))), (And(And(And(And(And(And(And(And(v["arg1_ndim"] == 4, Select(v["arg1_shape"], 0) > 0), Select(v["arg1_shape"], 1) > 0), Select(v["arg1_shape"], 2) > 0), Select(v["arg1_shape"], 3) > 0), 0 < Select(v["arg1_shape"], 1) * v["arg2_value"]), Select(v["arg1_shape"], 1) * v["arg2_value"] <= Select(v["arg1_shape"], 1)), 0 < Select(v["arg1_shape"], 2) * v["arg2_value"]), Select(v["arg1_shape"], 2) * v["arg2_value"] <= Select(v["arg1_shape"], 2))))))) if n else
          And(And(And(v["arg2_value"] > 0.0, v["arg2_value"] <= 1.0), (Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 6))), (Or((And(And(And(And(And(And(v["arg1_ndim"] == 3, Select(v["arg1_shape"], 0) > 0), Select(v["arg1_shape"], 1) > 0), 0 < Select(v["arg1_shape"], 0) * v["arg2_value"]), Select(v["arg1_shape"], 0) * v["arg2_value"] <= Select(v["arg1_shape"], 0)), 0 < Select(v["arg1_shape"], 1) * v["arg2_value"]), Select(v["arg1_shape"], 1) * v["arg2_value"] <= Select(v["arg1_shape"], 1))), (And(And(And(And(And(And(And(And(v["arg1_ndim"] == 4, Select(v["arg1_shape"], 0) > 0), Select(v["arg1_shape"], 1) > 0), Select(v["arg1_shape"], 2) > 0), Select(v["arg1_shape"], 3) > 0), 0 < Select(v["arg1_shape"], 1) * v["arg2_value"]), Select(v["arg1_shape"], 1) * v["arg2_value"] <= Select(v["arg1_shape"], 1)), 0 < Select(v["arg1_shape"], 2) * v["arg2_value"]), Select(v["arg1_shape"], 2) * v["arg2_value"] <= Select(v["arg1_shape"], 2)))))))
)

def rule_25_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Real('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)

        # Constraints for rule 25
        rule_25(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_25(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
