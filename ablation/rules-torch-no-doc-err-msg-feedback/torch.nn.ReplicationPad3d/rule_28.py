import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If padding mode is constant, the value should be within the valid range given the dtype. (Rule 28)

rule_28 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 21, If(v["arg3_dtype"] == 1, And(-128 <= v["arg2_value"], v["arg2_value"] <= 127), If(v["arg3_dtype"] == 2, And(-32768 <= v["arg2_value"], v["arg2_value"] <= 32767), True)), True)) if n else
          If(v["arg1_value"] == 21, If(v["arg3_dtype"] == 1, And(-128 <= v["arg2_value"], v["arg2_value"] <= 127), If(v["arg3_dtype"] == 2, And(-32768 <= v["arg2_value"], v["arg2_value"] <= 32767), True)), True))
)

def rule_28_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, (float, np.floating))):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_torch.index(arg1))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 28
        rule_28(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_28(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype']}, neg)
