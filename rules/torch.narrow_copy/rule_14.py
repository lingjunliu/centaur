import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If dim is a valid dimension index  (Rule 14)

rule_14 = lambda s, v, n=False: (
    s.add(Not(Or([And(i < (v["arg1_ndim"] - 1 + 1), Or(v["arg2_value"] == i, Or([And(i < (v["arg1_ndim"] - 1 + 1), v["arg2_value"] == (0 - (i + 1))) for i in range(6)]))) for i in range(6)])) if n else
          Or([And(i < (v["arg1_ndim"] - 1 + 1), Or(v["arg2_value"] == i, Or([And(i < (v["arg1_ndim"] - 1 + 1), v["arg2_value"] == (0 - (i + 1))) for i in range(6)]))) for i in range(6)]))
)

def rule_14_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 14
        rule_14(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_14(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
