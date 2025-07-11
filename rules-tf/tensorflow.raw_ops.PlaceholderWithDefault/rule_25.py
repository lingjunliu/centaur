import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the `shape` argument is provided and not an empty tuple, then the number of elements should be less than or equal to 8. (Rule 25)

rule_25 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_length"] > 0, v["arg1_length"] <= 8, False)) if n else
          If(v["arg1_length"] > 0, v["arg1_length"] <= 8, False))
)

def rule_25_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 25
        rule_25(solver, {'arg1_length': arg1_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_25(solver, {'arg1_length': arg1['length']}, neg)
