import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Check that if v_1 dtype is in the range, then max(v_2 (Rule 80)

rule_80 = lambda s, v, n=False: (
    s.add(Not(If(And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5), Select(v["arg2_range"], 1) < v["arg3_value"], False)) if n else
          If(And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5), Select(v["arg2_range"], 1) < v["arg3_value"], False))
)

def rule_80_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 80
        rule_80(solver, {'arg1_dtype': arg1_dtype, 'arg2_range': arg2_range, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_80(solver, {'arg1_dtype': arg1['dtype'], 'arg2_range': arg2['range'], 'arg3_value': arg3['value']}, neg)
