import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if the first tensor is a scalar with value zero, then the second tensor can only be zero, one or two dimensional. (Rule 78)

rule_78 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_ndim"] == 0, Select(v["arg1_range"], 1) == 0), Select(v["arg1_range"], 0) == 0), v["arg2_ndim"] <= 2, True)) if n else
          If(And(And(v["arg1_ndim"] == 0, Select(v["arg1_range"], 1) == 0), Select(v["arg1_range"], 0) == 0), v["arg2_ndim"] <= 2, True))
)

def rule_78_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 78
        rule_78(solver, {'arg1_ndim': arg1_ndim, 'arg1_range': arg1_range, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_78(solver, {'arg1_ndim': arg1['ndim'], 'arg1_range': arg1['range'], 'arg2_ndim': arg2['ndim']}, neg)
