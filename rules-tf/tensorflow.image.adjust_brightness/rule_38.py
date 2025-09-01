import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Image tensor must be of acceptable type (Rule 38)

rule_38 = lambda s, v, n=False: (
    s.add(Not(Or([And(x < (5 + 1), Or(v["arg1_dtype"] == x, Or([And(y < (9 + 1), v["arg1_dtype"] == y) for y in range(6)]))) for x in range(6)])) if n else
          Or([And(x < (5 + 1), Or(v["arg1_dtype"] == x, Or([And(y < (9 + 1), v["arg1_dtype"] == y) for y in range(6)]))) for x in range(6)]))
)

def rule_38_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 38
        rule_38(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_38(solver, {'arg1_dtype': arg1['dtype']}, neg)
