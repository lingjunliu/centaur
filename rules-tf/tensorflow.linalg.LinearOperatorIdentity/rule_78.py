import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If a tensor is provided, its last two dimensions should equal num_rows if batch_shape is None (Rule 78)

rule_78 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_length"] == 0, And(And(v["arg1_ndim"] >= 2, Select(v["arg1_shape"], v["arg1_ndim"] - 1) == v["arg2_value"]), Select(v["arg1_shape"], v["arg1_ndim"] - 2) == v["arg2_value"]), False)) if n else
          If(v["arg3_length"] == 0, And(And(v["arg1_ndim"] >= 2, Select(v["arg1_shape"], v["arg1_ndim"] - 1) == v["arg2_value"]), Select(v["arg1_shape"], v["arg1_ndim"] - 2) == v["arg2_value"]), False))
)

def rule_78_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_length = Int('arg3_length')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_length == len(arg3))

        # Constraints for rule 78
        rule_78(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_78(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_length': arg3['length']}, neg)
