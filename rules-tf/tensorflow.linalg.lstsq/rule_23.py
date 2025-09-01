import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# l2_regularizer and fast must have compatible shapes for non-trivial solutions (Rule 23)

rule_23 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == True, And(And(Select(v["arg3_shape"], v["arg3_ndim"] - 2) > 0, Select(v["arg3_shape"], v["arg3_ndim"] - 1) > 0), v["arg1_value"] >= 0), True)) if n else
          If(v["arg2_value"] == True, And(And(Select(v["arg3_shape"], v["arg3_ndim"] - 2) > 0, Select(v["arg3_shape"], v["arg3_ndim"] - 1) > 0), v["arg1_value"] >= 0), True))
)

def rule_23_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 23
        rule_23(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_ndim': arg3_ndim, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_23(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_ndim': arg3['ndim'], 'arg3_shape': arg3['shape']}, neg)
