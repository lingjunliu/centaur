import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If dtype of v_1 is complex64 and dtype of v_2 is complex128, then max of v_1 should be less than min of v_2 (Rule 95)

rule_95 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_dtype"] == 9, v["arg2_dtype"] == 10), Select(v["arg1_range"], 1) < Select(v["arg2_range"], 0), True)) if n else
          If(And(v["arg1_dtype"] == 9, v["arg2_dtype"] == 10), Select(v["arg1_range"], 1) < Select(v["arg2_range"], 0), True))
)

def rule_95_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 95
        rule_95(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype, 'arg2_range': arg2_range, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_95(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype'], 'arg2_range': arg2['range'], 'arg2_dtype': arg2['dtype']}, neg)
