import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# When include_batch_in_index is True, the indices in argmax should be less than or equal to batch * height * width * channels (Rule 61)

rule_61 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == True, Select(v["arg3_range"], 1) <= Select(v["arg2_shape"], 0) * Select(v["arg2_shape"], 1) * Select(v["arg2_shape"], 2) * Select(v["arg2_shape"], 3), True)) if n else
          If(v["arg1_value"] == True, Select(v["arg3_range"], 1) <= Select(v["arg2_shape"], 0) * Select(v["arg2_shape"], 1) * Select(v["arg2_shape"], 2) * Select(v["arg2_shape"], 3), True))
)

def rule_61_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_range = Array('arg3_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 61
        rule_61(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg3_range': arg3_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_61(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg3_range': arg3['range']}, neg)
