import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If x is scalar and dtype of a is float 32 or float64, then a's values can't be too large (Rule 73)

rule_73 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_ndim"] == 0, (Or(v["arg1_dtype"] == 8, v["arg1_dtype"] == 9))), Select(v["arg1_range"], 1) < 100000, False)) if n else
          If(And(v["arg2_ndim"] == 0, (Or(v["arg1_dtype"] == 8, v["arg1_dtype"] == 9))), Select(v["arg1_range"], 1) < 100000, False))
)

def rule_73_func(arg1, arg2, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 73
        rule_73(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_73(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range'], 'arg2_ndim': arg2['ndim']}, neg)
