import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The length of the shape must be less than or equal to 32 if specified as a tuple (Rule 54)

rule_54 = lambda s, v, n=False: (
    s.add(Not(v["arg1_length"] <= 32) if n else
          v["arg1_length"] <= 32)
)

def rule_54_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')

        # Value assignments
        solver.add(arg1_length == len(arg1))

        # Constraints for rule 54
        rule_54(solver, {'arg1_length': arg1_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_54(solver, {'arg1_length': arg1['length']}, neg)
