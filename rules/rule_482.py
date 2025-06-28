import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If string v_1 is one of the supported operations, tensor v_2 exists and contains at least 2 dimensions, then the shape values across dimension 0 and last dimension must be different (Rule 482)

rule_482 = lambda s, v, n=False: (
    s.add(Not(If(And((Or(Or(Or(Or(v["arg1_value"] == 7, v["arg1_value"] == 8), v["arg1_value"] == 9), v["arg1_value"] == 10), v["arg1_value"] == 11)), (v["arg2_ndim"] > 1)), Select(v["arg2_shape"], 0) != Select(v["arg2_shape"], v["arg2_ndim"] - 1), False)) if n else
          If(And((Or(Or(Or(Or(v["arg1_value"] == 7, v["arg1_value"] == 8), v["arg1_value"] == 9), v["arg1_value"] == 10), v["arg1_value"] == 11)), (v["arg2_ndim"] > 1)), Select(v["arg2_shape"], 0) != Select(v["arg2_shape"], v["arg2_ndim"] - 1), False))
)

def rule_482_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values.index(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 482
        rule_482(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_482(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']}, neg)
