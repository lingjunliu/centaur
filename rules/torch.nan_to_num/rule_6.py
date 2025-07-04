import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if input tensor's dtype is integer, then nan, posinf and neginf should also be integers if numbers (Rule 6)

rule_6 = lambda s, v, n=False: (
    s.add(Not(If(And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5), And(And((If(v["arg2_value"] != -10000000, And(1 <= v["arg2_value"], v["arg2_value"] <= 5), False)), (If(v["arg3_value"] != -10000000, And(1 <= v["arg3_value"], v["arg3_value"] <= 5), False))), (If(v["arg4_value"] != -10000000, And(1 <= v["arg4_value"], v["arg4_value"] <= 5), False))), False)) if n else
          If(And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5), And(And((If(v["arg2_value"] != -10000000, And(1 <= v["arg2_value"], v["arg2_value"] <= 5), False)), (If(v["arg3_value"] != -10000000, And(1 <= v["arg3_value"], v["arg3_value"] <= 5), False))), (If(v["arg4_value"] != -10000000, And(1 <= v["arg4_value"], v["arg4_value"] <= 5), False))), False))
)

def rule_6_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, (float, np.floating))):
            return False
        if not ((isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)) or isinstance(arg3, (float, np.floating))):
            return False
        if not ((isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)) or isinstance(arg4, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 6
        rule_6(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_6(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
