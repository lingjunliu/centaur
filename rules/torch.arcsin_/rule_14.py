import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Output must be complex if input is outside [-1,1] (Rule 14)

rule_14 = lambda s, v, n=False: (
    s.add(Not(If(Or(Select(v["arg1_range"], 0) < -1, Select(v["arg1_range"], 1) > 1), Or(v["arg2_value"] == 9, v["arg2_value"] == 10), False)) if n else
          If(Or(Select(v["arg1_range"], 0) < -1, Select(v["arg1_range"], 1) > 1), Or(v["arg2_value"] == 9, v["arg2_value"] == 10), False))
)

def rule_14_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))

        # Constraints for rule 14
        rule_14(solver, {'arg1_range': arg1_range, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_14(solver, {'arg1_range': arg1['range'], 'arg2_value': arg2['value']}, neg)
