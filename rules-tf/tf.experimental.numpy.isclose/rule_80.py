import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If tensors are both of float type and equal_nan is false, then atol and rtol must be different (Rule 80)

rule_80 = lambda s, v, n=False: (
    s.add(Not(If(And(And((And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8)), (And(6 <= v["arg2_dtype"], v["arg2_dtype"] <= 8))), v["arg5_value"] == False), v["arg3_value"] != v["arg4_value"], True)) if n else
          If(And(And((And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8)), (And(6 <= v["arg2_dtype"], v["arg2_dtype"] <= 8))), v["arg5_value"] == False), v["arg3_value"] != v["arg4_value"], True))
)

def rule_80_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False
        if not isinstance(arg4, (float, np.floating)):
            return False
        if not isinstance(arg5, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')
        arg3_value = Real('arg3_value')
        arg4_value = Real('arg4_value')
        arg5_value = Bool('arg5_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == arg4)
        solver.add(arg5_value == arg5)

        # Constraints for rule 80
        rule_80(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_80(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value']}, neg)
