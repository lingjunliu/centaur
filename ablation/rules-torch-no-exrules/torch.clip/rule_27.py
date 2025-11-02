import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Input tensor should not be float16 if min or max cannot be converted to half without overflow (Rule 27)

rule_27 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 7, And((Or([And(v_4 < (65504 + 1), v_4 == v["arg2_value"]) for v_4 in range(6)])), (Or([And(v_5 < (65504 + 1), v_5 == v["arg3_value"]) for v_5 in range(6)]))), True)) if n else
          If(v["arg1_dtype"] == 7, And((Or([And(v_4 < (65504 + 1), v_4 == v["arg2_value"]) for v_4 in range(6)])), (Or([And(v_5 < (65504 + 1), v_5 == v["arg3_value"]) for v_5 in range(6)]))), True))
)

def rule_27_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Real('arg2_value')
        arg3_value = Real('arg3_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)

        # Constraints for rule 27
        rule_27(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_27(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
