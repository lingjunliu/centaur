import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if x dtype is qint8, y dtype also qint8, min for x and y are less than 0 (Rule 42)

rule_42 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_dtype"] == 1, v["arg2_dtype"] == 1), Select(v["arg3_range"], 0) < 0, False)) if n else
          If(And(v["arg1_dtype"] == 1, v["arg2_dtype"] == 1), Select(v["arg3_range"], 0) < 0, False))
)

def rule_42_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')
        arg3_range = Array('arg3_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 42
        rule_42(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg3_range': arg3_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_42(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg3_range': arg3['range']}, neg)
