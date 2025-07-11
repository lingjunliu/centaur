import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if dtype is specified and is int or float, then starts, limits, and deltas should have the same dtype. (Rule 24)

rule_24 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg4_value"] != -1, (Or(Or(Or(Or(Or(Or(Or(v["arg4_value"] == 1, v["arg4_value"] == 2), v["arg4_value"] == 3), v["arg4_value"] == 4), v["arg4_value"] == 5), v["arg4_value"] == 6), v["arg4_value"] == 7), v["arg4_value"] == 8))), And(And(v["arg1_dtype"] == v["arg4_value"], v["arg2_dtype"] == v["arg4_value"]), v["arg3_dtype"] == v["arg4_value"]), False)) if n else
          If(And(v["arg4_value"] != -1, (Or(Or(Or(Or(Or(Or(Or(v["arg4_value"] == 1, v["arg4_value"] == 2), v["arg4_value"] == 3), v["arg4_value"] == 4), v["arg4_value"] == 5), v["arg4_value"] == 6), v["arg4_value"] == 7), v["arg4_value"] == 8))), And(And(v["arg1_dtype"] == v["arg4_value"], v["arg2_dtype"] == v["arg4_value"]), v["arg3_dtype"] == v["arg4_value"]), False))
)

def rule_24_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not (isinstance(arg4, torch.dtype) or isinstance(arg4, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')
        arg3_dtype = Int('arg3_dtype')
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_value == list_of_available_dtypes.index(np_dtype(arg4)))

        # Constraints for rule 24
        rule_24(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg3_dtype': arg3_dtype, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_24(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg3_dtype': arg3['dtype'], 'arg4_value': arg4['value']}, neg)
