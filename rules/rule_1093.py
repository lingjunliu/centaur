import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if the parameter mode is 'max' or 'sum' and the input type is fp16, then output_type must be fp32 (Rule 1093)

rule_1093 = lambda s, v, n=False: (
    s.add(Not(If(And((Or(v["arg3_value"] == 9, v["arg3_value"] == 8)), v["arg1_dtype"] == 6), v["arg2_value"] == 7, False)) if n else
          If(And((Or(v["arg3_value"] == 9, v["arg3_value"] == 8)), v["arg1_dtype"] == 6), v["arg2_value"] == 7, False))
)

def rule_1093_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False
        if not isinstance(arg3, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')
        arg3_value = String('arg3_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))
        solver.add(arg3_value == list_of_string_values.index(arg3))

        # Constraints for rule 1093
        rule_1093(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_1093(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
