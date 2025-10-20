import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if maximum of second tensor is less than 1, then dimension must be less than or equal 5 (Rule 57)

rule_57 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_range"], 1) < 1, v["arg1_ndim"] <= 5, True)) if n else
          If(Select(v["arg1_range"], 1) < 1, v["arg1_ndim"] <= 5, True))
)

def rule_57_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 57
        rule_57(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_57(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim']}, neg)
