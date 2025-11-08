import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the input tensor is of type half, then the bias alpha and beta should have low precision too (Rule 24)

rule_24 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 6, And(And(And(And(And(And(And(Select(v["arg1_range"], 0) >= -65504, Select(v["arg1_range"], 1) <= 65504), -65504 <= v["arg2_value"]), v["arg2_value"] <= 65504), -65504 <= v["arg3_value"]), v["arg3_value"] <= 65504), -65504 <= v["arg4_value"]), v["arg4_value"] <= 65504), True)) if n else
          If(v["arg1_dtype"] == 6, And(And(And(And(And(And(And(Select(v["arg1_range"], 0) >= -65504, Select(v["arg1_range"], 1) <= 65504), -65504 <= v["arg2_value"]), v["arg2_value"] <= 65504), -65504 <= v["arg3_value"]), v["arg3_value"] <= 65504), -65504 <= v["arg4_value"]), v["arg4_value"] <= 65504), True))
)

def rule_24_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False
        if not isinstance(arg4, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Real('arg2_value')
        arg3_value = Real('arg3_value')
        arg4_value = Real('arg4_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == arg4)

        # Constraints for rule 24
        rule_24(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_24(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
