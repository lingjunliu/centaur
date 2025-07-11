import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if init is true, then the data type should be a valid integer type (Rule 14)

rule_14 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == True, Or(Or(Or(Or(v["arg2_value"] == 1, v["arg2_value"] == 2), v["arg2_value"] == 3), v["arg2_value"] == 4), v["arg2_value"] == 5), False)) if n else
          If(v["arg1_value"] == True, Or(Or(Or(Or(v["arg2_value"] == 1, v["arg2_value"] == 2), v["arg2_value"] == 3), v["arg2_value"] == 4), v["arg2_value"] == 5), False))
)

def rule_14_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 14
        rule_14(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_14(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
