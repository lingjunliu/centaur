import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the ndim of b is 1, then the maximum value of b should be less than 5 (Rule 61)

rule_61 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 1, Select(v["arg1_range"], 1) < 5, True)) if n else
          If(v["arg1_ndim"] == 1, Select(v["arg1_range"], 1) < 5, True))
)

def rule_61_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 61
        rule_61(solver, {'arg1_ndim': arg1_ndim, 'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_61(solver, {'arg1_ndim': arg1['ndim'], 'arg1_range': arg1['range']}, neg)
