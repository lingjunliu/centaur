import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if dimension is None, edge_order must be 1 or 2. (Rule 24)

rule_24 = lambda s, v, n=False: (
    s.add(Not(Or((And([Implies(i < (0 + 1), v["arg1_value"] == 0.0) for i in range(6)])), (Or(v["arg2_value"] == 1, v["arg2_value"] == 2)))) if n else
          Or((And([Implies(i < (0 + 1), v["arg1_value"] == 0.0) for i in range(6)])), (Or(v["arg2_value"] == 1, v["arg2_value"] == 2))))
)

def rule_24_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 24
        rule_24(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_24(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
