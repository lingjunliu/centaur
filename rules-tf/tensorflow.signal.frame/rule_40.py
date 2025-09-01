import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If signal dimension along axis is zero and pad_end is True, then frame_length and frame_step are not used and pad_value data type should match (Rule 40)

rule_40 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_shape"], v["arg2_value"]) == 0, And(v["arg3_value"] == True, v["arg1_dtype"] == v["arg4_dtype"]), True)) if n else
          If(Select(v["arg1_shape"], v["arg2_value"]) == 0, And(v["arg3_value"] == True, v["arg1_dtype"] == v["arg4_dtype"]), True))
)

def rule_40_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        if not isinstance(arg3, bool):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')
        arg3_value = Bool('arg3_value')
        arg4_dtype = Int('arg4_dtype')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == arg3)
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))

        # Constraints for rule 40
        rule_40(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_dtype': arg4_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_40(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_dtype': arg4['dtype']}, neg)
