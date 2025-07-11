import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the input tensor's type is not float16, float32 or bfloat16 then depth_radius must be zero and alpha must be 0 and beta must be 1 and bias must be 1. (Rule 128)

rule_128 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_dtype"] != 6, v["arg1_dtype"] != 7), v["arg1_dtype"] != 8), And(And(And(v["arg2_value"] == 0, v["arg3_value"] == 0), v["arg4_value"] == 1), v["arg5_value"] == 1), False)) if n else
          If(And(And(v["arg1_dtype"] != 6, v["arg1_dtype"] != 7), v["arg1_dtype"] != 8), And(And(And(v["arg2_value"] == 0, v["arg3_value"] == 0), v["arg4_value"] == 1), v["arg5_value"] == 1), False))
)

def rule_128_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False
        if not isinstance(arg4, (float, np.floating)):
            return False
        if not isinstance(arg5, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')
        arg3_value = Real('arg3_value')
        arg4_value = Real('arg4_value')
        arg5_value = Real('arg5_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == arg4)
        solver.add(arg5_value == arg5)

        # Constraints for rule 128
        rule_128(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_128(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value']}, neg)
