import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# features tensor dtype must be an element of the allowed types (Rule 10)

rule_10 = lambda s, v, n=False: (
    s.add(Not(Or([And(t < (13 + 1), And((Or(Or(Or(Or(Or(Or(Or(Or(Or(t == 1, t == 2), t == 3), t == 4), t == 5), t == 6), t == 7), t == 8), t == 9), t == 13)), v["arg1_dtype"] == t)) for t in range(6)])) if n else
          Or([And(t < (13 + 1), And((Or(Or(Or(Or(Or(Or(Or(Or(Or(t == 1, t == 2), t == 3), t == 4), t == 5), t == 6), t == 7), t == 8), t == 9), t == 13)), v["arg1_dtype"] == t)) for t in range(6)]))
)

def rule_10_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 10
        rule_10(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_10(solver, {'arg1_dtype': arg1['dtype']}, neg)
