import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If v1 has more dimensions than v2, then the minimum value of v1 must be smaller than the maximum value of v2 (Rule 98)

rule_98 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] > v["arg2_ndim"], Select(v["arg1_range"], 0) < Select(v["arg2_range"], 1), True)) if n else
          If(v["arg1_ndim"] > v["arg2_ndim"], Select(v["arg1_range"], 0) < Select(v["arg2_range"], 1), True))
)

def rule_98_func(arg1, arg2, solver=None, neg=False):
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
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_ndim == arg2.ndim)
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 98
        rule_98(solver, {'arg1_ndim': arg1_ndim, 'arg1_range': arg1_range, 'arg2_ndim': arg2_ndim, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_98(solver, {'arg1_ndim': arg1['ndim'], 'arg1_range': arg1['range'], 'arg2_ndim': arg2['ndim'], 'arg2_range': arg2['range']}, neg)
