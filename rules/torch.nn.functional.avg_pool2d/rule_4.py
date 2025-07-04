import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# pad should be at most half of effective kernel size. Suppresses "pad should be at most half of effective kernel size, but got pad=0, kernel_size=-2 and dilation=1". This rule only handles the case where kernel_size and pad are integers (Rule 4)

rule_4 = lambda s, v, n=False: (
    s.add(Not(v["arg1_value"] >= 2 * v["arg2_value"]) if n else
          v["arg1_value"] >= 2 * v["arg2_value"])
)

def rule_4_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 4
        rule_4(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_4(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
