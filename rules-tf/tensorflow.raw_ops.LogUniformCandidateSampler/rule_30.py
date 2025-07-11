import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# num_true, num_sampled, and range_max must all be within a reasonable range (e.g., less than 2^30 (Rule 30)

rule_30 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_value"] < 1073741824, v["arg2_value"] < 1073741824), v["arg3_value"] < 1073741824)) if n else
          And(And(v["arg1_value"] < 1073741824, v["arg2_value"] < 1073741824), v["arg3_value"] < 1073741824))
)

def rule_30_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 30
        rule_30(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_30(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
