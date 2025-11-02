import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Minimum element in tensor v_1 must be greater than or equal to v_2 and the maximum element must be less than or equal to v_3. (Rule 74)

rule_74 = lambda s, v, n=False: (
    s.add(Not(And(Select(v["arg1_range"], 0) >= v["arg2_value"], Select(v["arg1_range"], 1) <= v["arg3_value"])) if n else
          And(Select(v["arg1_range"], 0) >= v["arg2_value"], Select(v["arg1_range"], 1) <= v["arg3_value"]))
)

def rule_74_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, (float, np.floating))):
            return False
        if not ((isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)) or isinstance(arg3, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 74
        rule_74(solver, {'arg1_range': arg1_range, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_74(solver, {'arg1_range': arg1['range'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
