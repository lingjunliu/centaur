import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If align_corners is true then the shapes of images in dimension 1 and 2 must be greater than 1 and the elements of size must be int32 (Rule 37)

rule_37 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] == True, And(And(Select(v["arg1_shape"], 1) > 1, Select(v["arg1_shape"], 2) > 1), v["arg2_dtype"] == 3), True)) if n else
          If(v["arg3_value"] == True, And(And(Select(v["arg1_shape"], 1) > 1, Select(v["arg1_shape"], 2) > 1), v["arg2_dtype"] == 3), True))
)

def rule_37_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg3_value = Bool('arg3_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_value == arg3)

        # Constraints for rule 37
        rule_37(solver, {'arg1_shape': arg1_shape, 'arg2_dtype': arg2_dtype, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_37(solver, {'arg1_shape': arg1['shape'], 'arg2_dtype': arg2['dtype'], 'arg3_value': arg3['value']}, neg)
