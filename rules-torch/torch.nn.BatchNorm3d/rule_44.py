import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Number of Features should be a power of 2 for performance reasons (Rule 44)

rule_44 = lambda s, v, n=False: (
    s.add(Not(Or([And(i < (12 + 1), v["arg1_value"] == 2 * i) for i in range(6)])) if n else
          Or([And(i < (12 + 1), v["arg1_value"] == 2 * i) for i in range(6)]))
)

def rule_44_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 44
        rule_44(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_44(solver, {'arg1_value': arg1['value']}, neg)
