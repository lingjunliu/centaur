import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If dim is invalid and no out, the check passes - preventing invalid dim (Rule 120)

rule_120 = lambda s, v, n=False: (
    s.add(Not(If(And(Or((0 - v["arg1_ndim"]) > v["arg2_value"], v["arg2_value"] >= v["arg1_ndim"]), Select(v["arg3_shape"], 0) == 0), True, True)) if n else
          If(And(Or((0 - v["arg1_ndim"]) > v["arg2_value"], v["arg2_value"] >= v["arg1_ndim"]), Select(v["arg3_shape"], 0) == 0), True, True))
)

def rule_120_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = Int('arg2_value')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == int(arg2))
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 120
        rule_120(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_120(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape']}, neg)
