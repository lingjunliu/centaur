import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the dimension of X1 is less than 2 and X2 is a complex, then it's max should be less than 50 (Rule 46)

rule_46 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_ndim"] < 2, (Or(v["arg2_dtype"] == 10, v["arg2_dtype"] == 11))), Select(v["arg2_range"], 1) < 50, True)) if n else
          If(And(v["arg1_ndim"] < 2, (Or(v["arg2_dtype"] == 10, v["arg2_dtype"] == 11))), Select(v["arg2_range"], 1) < 50, True))
)

def rule_46_func(arg1, arg2, solver=None, neg=False):
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
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 46
        rule_46(solver, {'arg1_ndim': arg1_ndim, 'arg2_range': arg2_range, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_46(solver, {'arg1_ndim': arg1['ndim'], 'arg2_range': arg2['range'], 'arg2_dtype': arg2['dtype']}, neg)
