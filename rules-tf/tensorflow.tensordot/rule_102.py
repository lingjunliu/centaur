import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If axes is a scalar N, it must be representable as int32 (Rule 102)

rule_102 = lambda s, v, n=False: (
    s.add(Not(And(-2147483648 <= v["arg1_value"], v["arg1_value"] <= 2147483647)) if n else
          And(-2147483648 <= v["arg1_value"], v["arg1_value"] <= 2147483647))
)

def rule_102_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))

        # Constraints for rule 102
        rule_102(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_102(solver, {'arg1_value': arg1['value']}, neg)
