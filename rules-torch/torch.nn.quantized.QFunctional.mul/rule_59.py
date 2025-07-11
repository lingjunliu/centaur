import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If scale is specified then make sure parameters are properly restricted (Rule 59)

rule_59 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] == 1, And((And(v["arg2_value"] >= -128, v["arg2_value"] <= 127)), (v["arg1_value"] > 0)), If(v["arg3_value"] == 5, And((And(v["arg2_value"] >= 0, v["arg2_value"] <= 255)), (v["arg1_value"] > 0)), False))) if n else
          If(v["arg3_value"] == 1, And((And(v["arg2_value"] >= -128, v["arg2_value"] <= 127)), (v["arg1_value"] > 0)), If(v["arg3_value"] == 5, And((And(v["arg2_value"] >= 0, v["arg2_value"] <= 255)), (v["arg1_value"] > 0)), False)))
)

def rule_59_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, torch.dtype) or isinstance(arg3, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == list_of_available_dtypes.index(np_dtype(arg3)))

        # Constraints for rule 59
        rule_59(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_59(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
