import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

#  If replacement is not valid for encoding then strict is used, errors is set, or ignore. (Rule 99)

rule_99 = lambda s, v, n=False: (
    s.add(Not(If(Or(Select(v["arg1_range"], 0) < 0, Select(v["arg1_range"], 1) > 1114111), Or(Or(v["arg2_value"] == 45, v["arg2_value"] == 44), v["arg2_value"] == 43), True)) if n else
          If(Or(Select(v["arg1_range"], 0) < 0, Select(v["arg1_range"], 1) > 1114111), Or(Or(v["arg2_value"] == 45, v["arg2_value"] == 44), v["arg2_value"] == 43), True))
)

def rule_99_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))

        # Constraints for rule 99
        rule_99(solver, {'arg1_range': arg1_range, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_99(solver, {'arg1_range': arg1['range'], 'arg2_value': arg2['value']}, neg)
