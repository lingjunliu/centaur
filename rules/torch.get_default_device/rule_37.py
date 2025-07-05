import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If the device is specified as an integer, it must be a valid CUDA device index if CUDA is available. (Rule 37)

rule_37 = lambda s, v, n=False: (
    s.add(Not(If(Or([And(i < (7 + 1), i == v["arg1_value"]) for i in range(6)]), True, False)) if n else
          If(Or([And(i < (7 + 1), i == v["arg1_value"]) for i in range(6)]), True, False))
)

def rule_37_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 37
        rule_37(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_37(solver, {'arg1_value': arg1['value']}, neg)
