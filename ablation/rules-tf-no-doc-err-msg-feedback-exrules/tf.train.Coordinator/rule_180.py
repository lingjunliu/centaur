import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# When a thread writes output, no race conditions will arise and write will be successfully committed to a storage. (Rule 180)

rule_180 = lambda s, v, n=False: (
    s.add(Not(v["arg1_value"] == True) if n else
          v["arg1_value"] == True)
)

def rule_180_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')

        # Value assignments
        solver.add(arg1_value == arg1)

        # Constraints for rule 180
        rule_180(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_180(solver, {'arg1_value': arg1['value']}, neg)
