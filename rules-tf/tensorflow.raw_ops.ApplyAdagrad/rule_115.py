import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if lr is 0, then all values in grad must be zero too, or accum and grad must be the same type (Rule 115)

rule_115 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_range"], 1) == 0, Or((And(Select(v["arg2_range"], 0) == 0, Select(v["arg2_range"], 1) == 0)), v["arg2_dtype"] == v["arg3_dtype"]), False)) if n else
          If(Select(v["arg1_range"], 1) == 0, Or((And(Select(v["arg2_range"], 0) == 0, Select(v["arg2_range"], 1) == 0)), v["arg2_dtype"] == v["arg3_dtype"]), False))
)

def rule_115_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 115
        rule_115(solver, {'arg1_range': arg1_range, 'arg2_dtype': arg2_dtype, 'arg2_range': arg2_range, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_115(solver, {'arg1_range': arg1['range'], 'arg2_dtype': arg2['dtype'], 'arg2_range': arg2['range'], 'arg3_dtype': arg3['dtype']}, neg)
