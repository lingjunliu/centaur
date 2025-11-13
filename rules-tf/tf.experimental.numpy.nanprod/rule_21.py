import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the axis is None, then keepdims doesn't affect the dimension (Rule 21)

rule_21 = lambda s, v, n=False: (
    s.add(Not(If(axis == -1, Or(v["arg2_value"] == True, v["arg2_value"] == False), True)) if n else
          If(axis == -1, Or(v["arg2_value"] == True, v["arg2_value"] == False), True))
)

def rule_21_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg2_value = Bool('arg2_value')

        # Value assignments
        solver.add(arg2_value == arg2)

        # Constraints for rule 21
        rule_21(solver, {'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_21(solver, {'arg2_value': arg2['value']}, neg)
