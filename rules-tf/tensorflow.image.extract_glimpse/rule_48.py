import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# offsets tensor values should be within valid range based on normalization (Rule 48)

rule_48 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == True, (And(Select(v["arg1_range"], 0) >= -1, Select(v["arg1_range"], 1) <= 1)), (And(Select(v["arg1_range"], 0) >= 0, Select(v["arg1_range"], 1) <= (If(Select(v["arg3_shape"], 1) < Select(v["arg3_shape"], 2), Select(v["arg3_shape"], 1), Select(v["arg3_shape"], 2))))))) if n else
          If(v["arg2_value"] == True, (And(Select(v["arg1_range"], 0) >= -1, Select(v["arg1_range"], 1) <= 1)), (And(Select(v["arg1_range"], 0) >= 0, Select(v["arg1_range"], 1) <= (If(Select(v["arg3_shape"], 1) < Select(v["arg3_shape"], 2), Select(v["arg3_shape"], 1), Select(v["arg3_shape"], 2)))))))
)

def rule_48_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Bool('arg2_value')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == arg2)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 48
        rule_48(solver, {'arg1_range': arg1_range, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_48(solver, {'arg1_range': arg1['range'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape']}, neg)
