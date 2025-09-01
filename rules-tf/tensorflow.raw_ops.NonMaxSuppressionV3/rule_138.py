import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# iou_threshold should be between 0 and 1 and score_threshold between -1 and 1, boxes have same dtype as iou_threshold (Rule 138)

rule_138 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(Select(v["arg1_range"], 0) >= 0, Select(v["arg1_range"], 1) <= 1), Select(v["arg2_range"], 0) >= -1), Select(v["arg2_range"], 1) <= 1), v["arg3_dtype"] == v["arg1_dtype"])) if n else
          And(And(And(And(Select(v["arg1_range"], 0) >= 0, Select(v["arg1_range"], 1) <= 1), Select(v["arg2_range"], 0) >= -1), Select(v["arg2_range"], 1) <= 1), v["arg3_dtype"] == v["arg1_dtype"]))
)

def rule_138_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 138
        rule_138(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype, 'arg2_range': arg2_range, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_138(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype'], 'arg2_range': arg2['range'], 'arg3_dtype': arg3['dtype']}, neg)
