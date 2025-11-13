import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# strides should be of length 1, 2 or N (N spatial dims (Rule 4)

rule_4 = lambda s, v, n=False: (
    s.add(Not(Or(Or(v["arg1_length"] == 1, v["arg1_length"] == 2), v["arg1_length"] == v["arg2_value"])) if n else
          Or(Or(v["arg1_length"] == 1, v["arg1_length"] == 2), v["arg1_length"] == v["arg2_value"]))
)

def rule_4_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 4
        rule_4(solver, {'arg1_length': arg1_length, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_4(solver, {'arg1_length': arg1['length'], 'arg2_value': arg2['value']}, neg)
