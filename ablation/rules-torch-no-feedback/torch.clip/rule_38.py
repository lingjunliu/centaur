import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If output is specified and input is uint8, then min and max, if provided, must also be within the uint8 range (Rule 38)

rule_38 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 6, (And(And(And(And(v["arg4_dtype"] == 6, 0 <= v["arg2_value"]), v["arg2_value"] <= 255), 0 <= v["arg3_value"]), v["arg3_value"] <= 255)), True)) if n else
          If(v["arg1_dtype"] == 6, (And(And(And(And(v["arg4_dtype"] == 6, 0 <= v["arg2_value"]), v["arg2_value"] <= 255), 0 <= v["arg3_value"]), v["arg3_value"] <= 255)), True))
)

def rule_38_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Real('arg2_value')
        arg3_value = Real('arg3_value')
        arg4_dtype = Int('arg4_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))

        # Constraints for rule 38
        rule_38(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_dtype': arg4_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_38(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_dtype': arg4['dtype']}, neg)
