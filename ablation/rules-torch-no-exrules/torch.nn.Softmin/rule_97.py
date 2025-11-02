import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The dimension has to be within the valid range (Rule 97)

rule_97 = lambda s, v, n=False: (
    s.add(Not(Or([And(i < ((v["arg1_ndim"] - 1) + 1), v["arg2_value"] == i) for i in range(6)])) if n else
          Or([And(i < ((v["arg1_ndim"] - 1) + 1), v["arg2_value"] == i) for i in range(6)]))
)

def rule_97_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 97
        rule_97(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_97(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
