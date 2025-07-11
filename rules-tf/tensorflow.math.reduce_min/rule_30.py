import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the first tensor's dimension is less than the second's then the first tensor's maximum value should be less than the second tensor's dimension  (Rule 30)

rule_30 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] < v["arg2_ndim"], Select(v["arg1_range"], 1) < v["arg2_ndim"], False)) if n else
          If(v["arg1_ndim"] < v["arg2_ndim"], Select(v["arg1_range"], 1) < v["arg2_ndim"], False))
)

def rule_30_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 30
        rule_30(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_30(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim']}, neg)
