import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Check the data type of the tensor and if it's smaller than a certain number OR equal to a number and the size is smaller than a number and larger than 0 (Rule 74)

rule_74 = lambda s, v, n=False: (
    s.add(Not(And(And((Or((v["arg1_dtype"] < v["arg2_value"]), (v["arg1_dtype"] == v["arg3_value"]))), Select(v["arg1_shape"], 0) < v["arg4_value"]), Select(v["arg1_shape"], 0) > 0)) if n else
          And(And((Or((v["arg1_dtype"] < v["arg2_value"]), (v["arg1_dtype"] == v["arg3_value"]))), Select(v["arg1_shape"], 0) < v["arg4_value"]), Select(v["arg1_shape"], 0) > 0))
)

def rule_74_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == int(arg4))

        # Constraints for rule 74
        rule_74(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_74(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
