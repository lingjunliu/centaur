import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If v_2 is a 3D tensor, then the min of v_1 must be positive  (Rule 52)

rule_52 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_ndim"] == 3, Select(v["arg1_range"], 0) > 0, True)) if n else
          If(v["arg2_ndim"] == 3, Select(v["arg1_range"], 0) > 0, True))
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
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 52
        rule_52(solver, {'arg1_range': arg1_range, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_52(solver, {'arg1_range': arg1['range'], 'arg2_ndim': arg2['ndim']}, neg)
