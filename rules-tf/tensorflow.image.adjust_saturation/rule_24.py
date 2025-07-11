import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If saturation factor is significantly large, the output is capped at maximum value possible for that dtype (Rule 24)

rule_24 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] > 1000, Select(v["arg3_range"], 1) == (If(v["arg1_dtype"] == 1, 127, (If(v["arg1_dtype"] == 2, 32767, (If(v["arg1_dtype"] == 3, 2147483647, (If(v["arg1_dtype"] == 4, 9223372036854775807, (If(v["arg1_dtype"] == 5, 255, (If(v["arg1_dtype"] == 6, 65500, 1000000000)))))))))))), False)) if n else
          If(v["arg2_value"] > 1000, Select(v["arg3_range"], 1) == (If(v["arg1_dtype"] == 1, 127, (If(v["arg1_dtype"] == 2, 32767, (If(v["arg1_dtype"] == 3, 2147483647, (If(v["arg1_dtype"] == 4, 9223372036854775807, (If(v["arg1_dtype"] == 5, 255, (If(v["arg1_dtype"] == 6, 65500, 1000000000)))))))))))), False))
)

def rule_24_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Real('arg2_value')
        arg3_range = Array('arg3_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 24
        rule_24(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_range': arg3_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_24(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_range': arg3['range']}, neg)
