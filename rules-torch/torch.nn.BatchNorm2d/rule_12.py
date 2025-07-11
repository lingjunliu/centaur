import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# num_features must be a concrete integer to avoid SymIntArrayRef error (Rule 12)

rule_12 = lambda s, v, n=False: (
    s.add(Not(Or([And(x < (100 + 1), Or(v["arg1_value"] == x, Or([And(x < (1000000 + 1), v["arg1_value"] == x) for x in range(6)]))) for x in range(6)])) if n else
          Or([And(x < (100 + 1), Or(v["arg1_value"] == x, Or([And(x < (1000000 + 1), v["arg1_value"] == x) for x in range(6)]))) for x in range(6)]))
)

def rule_12_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 12
        rule_12(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_12(solver, {'arg1_value': arg1['value']}, neg)
