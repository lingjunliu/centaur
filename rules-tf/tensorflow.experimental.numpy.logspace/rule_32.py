import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If a Dtype is provided it must match start and stop if they are tensors. Start and stop have to be the same tensor type, and not strings. num > 0, base > 0 (Rule 32)

rule_32 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] != v["arg3_value"], And(And(And(And(v["arg2_dtype"] == v["arg3_value"], v["arg1_dtype"] == v["arg2_dtype"]), v["arg1_dtype"] != 12), v["arg4_value"] > 0), v["arg5_value"] > 0), False)) if n else
          If(v["arg1_dtype"] != v["arg3_value"], And(And(And(And(v["arg2_dtype"] == v["arg3_value"], v["arg1_dtype"] == v["arg2_dtype"]), v["arg1_dtype"] != 12), v["arg4_value"] > 0), v["arg5_value"] > 0), False))
)

def rule_32_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, torch.dtype) or isinstance(arg3, tf.dtypes.DType)):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False
        if not isinstance(arg5, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')
        arg5_value = Real('arg5_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_value == list_of_available_dtypes.index(np_dtype(arg3)))
        solver.add(arg4_value == int(arg4))
        solver.add(arg5_value == arg5)

        # Constraints for rule 32
        rule_32(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_32(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value']}, neg)
