import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If params has more than 4 dimension, then the indices values should be less than 10 (Rule 52)

rule_52 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] > 4, Select(v["arg2_range"], 1) < 10, False)) if n else
          If(v["arg1_ndim"] > 4, Select(v["arg2_range"], 1) < 10, False))
)

def rule_52_func(arg1, arg2, solver=None, neg=False):
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
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 52
        rule_52(solver, {'arg1_ndim': arg1_ndim, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_52(solver, {'arg1_ndim': arg1['ndim'], 'arg2_range': arg2['range']}, neg)
