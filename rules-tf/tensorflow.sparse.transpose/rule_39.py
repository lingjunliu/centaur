import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the length of perm is specified, it must be greater than zero (Rule 39)

rule_39 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_length"] > 0, v["arg1_length"] > 0, False)) if n else
          If(v["arg1_length"] > 0, v["arg1_length"] > 0, False))
)

def rule_39_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')

        # Value assignments
        solver.add(arg1_length == len(arg1))

        # Constraints for rule 39
        rule_39(solver, {'arg1_length': arg1_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_39(solver, {'arg1_length': arg1['length']}, neg)
