import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Minimum value must be lower than a value, if tensor has float values and string is linear  (Rule 93)

rule_93 = lambda s, v, n=False: (
    s.add(Not(If(And((Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8)), v["arg3_value"] == 20), Select(v["arg1_range"], 0) <= v["arg2_value"], True)) if n else
          If(And((Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8)), v["arg3_value"] == 20), Select(v["arg1_range"], 0) <= v["arg2_value"], True))
)

def rule_93_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Real('arg2_value')
        arg3_value = String('arg3_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == list_of_string_values_torch.index(arg3))

        # Constraints for rule 93
        rule_93(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_93(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
