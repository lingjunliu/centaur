import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if validate_indices is true, then if a is sparse then a must have indices sorted in row major order (Rule 79)

rule_79 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == True, True, True)) if n else
          If(v["arg2_value"] == True, True, True))
)

def rule_79_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 79
        rule_79(solver, {'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_79(solver, {'arg2_value': arg2['value']}, neg)
