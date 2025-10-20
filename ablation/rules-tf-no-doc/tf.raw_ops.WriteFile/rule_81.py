import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the max is bigger than 5, the number of dimension must be less than 3 and minimum value must be bigger than 2 (Rule 81)

rule_81 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_range"], 1) > 5, And(v["arg1_ndim"] < 3, Select(v["arg1_range"], 0) > 2), True)) if n else
          If(Select(v["arg1_range"], 1) > 5, And(v["arg1_ndim"] < 3, Select(v["arg1_range"], 0) > 2), True))
)

def rule_81_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 81
        rule_81(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_81(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim']}, neg)
