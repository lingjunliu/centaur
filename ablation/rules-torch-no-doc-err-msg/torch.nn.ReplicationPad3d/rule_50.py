import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Padding value if mode is constant should be within the range of the tensor's dtype (Rule 50)

rule_50 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] == 21, If(v["arg1_dtype"] == 6, And(-65504 <= v["arg2_value"], v["arg2_value"] <= 65504), If(v["arg1_dtype"] == 7, And(-3.4028235e+38 <= v["arg2_value"], v["arg2_value"] <= 3.4028235e+38), If(v["arg1_dtype"] == 8, And(-1.7976931348623157e+308 <= v["arg2_value"], v["arg2_value"] <= 1.7976931348623157e+308), If(v["arg1_dtype"] == 9, True, If(v["arg1_dtype"] == 10, True, True))))), True)) if n else
          If(v["arg3_value"] == 21, If(v["arg1_dtype"] == 6, And(-65504 <= v["arg2_value"], v["arg2_value"] <= 65504), If(v["arg1_dtype"] == 7, And(-3.4028235e+38 <= v["arg2_value"], v["arg2_value"] <= 3.4028235e+38), If(v["arg1_dtype"] == 8, And(-1.7976931348623157e+308 <= v["arg2_value"], v["arg2_value"] <= 1.7976931348623157e+308), If(v["arg1_dtype"] == 9, True, If(v["arg1_dtype"] == 10, True, True))))), True))
)

def rule_50_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Real('arg2_value')
        arg3_value = String('arg3_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == list_of_string_values_torch.index(arg3))

        # Constraints for rule 50
        rule_50(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_50(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
