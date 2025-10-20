import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If no axis is specified, and input tensor dtype is integer with size smaller than int32, the output dtype should be at least int32 if no dtype is given (Rule 86)

rule_86 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == -666, (If((Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2)), Or(v["arg3_value"] == -666, v["arg3_value"] >= 3), True)), True)) if n else
          If(v["arg2_value"] == -666, (If((Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2)), Or(v["arg3_value"] == -666, v["arg3_value"] >= 3), True)), True))
)

def rule_86_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, torch.dtype) or isinstance(arg3, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == list_of_available_dtypes.index(np_dtype(arg3)))

        # Constraints for rule 86
        rule_86(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_86(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
