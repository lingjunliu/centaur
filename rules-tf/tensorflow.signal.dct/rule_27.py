import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if n is greater than 0, type must be 2,3, or 4 (Rule 27)

rule_27 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] > 0, Or(Or(v["arg2_value"] == 2, v["arg2_value"] == 3), v["arg2_value"] == 4), False)) if n else
          If(v["arg1_value"] > 0, Or(Or(v["arg2_value"] == 2, v["arg2_value"] == 3), v["arg2_value"] == 4), False))
)

def rule_27_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 27
        rule_27(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_27(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
