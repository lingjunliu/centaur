import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If min is specified and max is specified and input tensor is of integral type, then if output tensor is specified, its type is also integral and min and max are also appropriate to that type (Rule 32)

rule_32 = lambda s, v, n=False: (
    s.add(Not(If(And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5), (And(And(And(1 <= v["arg4_dtype"], v["arg4_dtype"] <= 5), v["arg2_value"] % 1 == 0), v["arg3_value"] % 1 == 0)), True)) if n else
          If(And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5), (And(And(And(1 <= v["arg4_dtype"], v["arg4_dtype"] <= 5), v["arg2_value"] % 1 == 0), v["arg3_value"] % 1 == 0)), True))
)

def rule_32_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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

        # Constraints for rule 32
        rule_32(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_dtype': arg4_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_32(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_dtype': arg4['dtype']}, neg)
