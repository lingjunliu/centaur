import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The values of lower and upper should make sense for the selected dtype (Rule 36)

rule_36 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_dtype"] == 1, And(v["arg1_value"] > -128, v["arg2_value"] < 127), If(v["arg3_dtype"] == 2, And(v["arg1_value"] > -32768, v["arg2_value"] < 32767), If(v["arg3_dtype"] == 3, And(v["arg1_value"] > -2147483648, v["arg2_value"] < 2147483647), True)))) if n else
          If(v["arg3_dtype"] == 1, And(v["arg1_value"] > -128, v["arg2_value"] < 127), If(v["arg3_dtype"] == 2, And(v["arg1_value"] > -32768, v["arg2_value"] < 32767), If(v["arg3_dtype"] == 3, And(v["arg1_value"] > -2147483648, v["arg2_value"] < 2147483647), True))))
)

def rule_36_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Real('arg2_value')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 36
        rule_36(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_36(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype']}, neg)
