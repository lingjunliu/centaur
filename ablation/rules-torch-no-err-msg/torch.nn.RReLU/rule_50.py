import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If inplace is true then lower and upper must be in a valid range for float16, float32, float64 (Rule 50)

rule_50 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] == True, If(v["arg4_dtype"] == 6, And(v["arg1_value"] > -65504, v["arg2_value"] < 65504), If(v["arg4_dtype"] == 7, And(v["arg1_value"] > -3.4028235e+38, v["arg2_value"] < 3.4028235e+38), If(v["arg4_dtype"] == 8, And(v["arg1_value"] > -1.7976931348623157e+308, v["arg2_value"] < 1.7976931348623157e+308), True))), True)) if n else
          If(v["arg3_value"] == True, If(v["arg4_dtype"] == 6, And(v["arg1_value"] > -65504, v["arg2_value"] < 65504), If(v["arg4_dtype"] == 7, And(v["arg1_value"] > -3.4028235e+38, v["arg2_value"] < 3.4028235e+38), If(v["arg4_dtype"] == 8, And(v["arg1_value"] > -1.7976931348623157e+308, v["arg2_value"] < 1.7976931348623157e+308), True))), True))
)

def rule_50_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, bool):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Real('arg2_value')
        arg3_value = Bool('arg3_value')
        arg4_dtype = Int('arg4_dtype')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))

        # Constraints for rule 50
        rule_50(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_dtype': arg4_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_50(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_dtype': arg4['dtype']}, neg)
