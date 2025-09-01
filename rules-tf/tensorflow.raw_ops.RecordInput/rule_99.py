import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If file_buffer_size is greater than 100000 then file_shuffle_shift_ratio must be less than 0.1 (Rule 99)

rule_99 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] > 100000, v["arg2_value"] < 0.1, True)) if n else
          If(v["arg1_value"] > 100000, v["arg2_value"] < 0.1, True))
)

def rule_99_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Real('arg2_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == arg2)

        # Constraints for rule 99
        rule_99(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_99(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
