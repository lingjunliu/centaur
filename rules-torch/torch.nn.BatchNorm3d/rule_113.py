import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If you don't use affine, you need to track the running stats to prevent 'Bool' dtype error, when using small tensors. (Rule 113)

rule_113 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == False, Select(v["arg2_shape"], 1) * Select(v["arg2_shape"], 2) * Select(v["arg2_shape"], 3) * Select(v["arg2_shape"], 4) < 1000), v["arg3_value"] == True, False)) if n else
          If(And(v["arg1_value"] == False, Select(v["arg2_shape"], 1) * Select(v["arg2_shape"], 2) * Select(v["arg2_shape"], 3) * Select(v["arg2_shape"], 4) < 1000), v["arg3_value"] == True, False))
)

def rule_113_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == arg3)

        # Constraints for rule 113
        rule_113(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_113(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value']}, neg)
