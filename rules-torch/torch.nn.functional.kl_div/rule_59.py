import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if the target tensor has any value equal to 0, input tensor value must be -inf (Rule 59)

rule_59 = lambda s, v, n=False: (
    s.add(Not(If((Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_range"], 0) == 0) for i in range(6)])), Select(v["arg1_range"], 0) == -10000000000000000.0, False)) if n else
          If((Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_range"], 0) == 0) for i in range(6)])), Select(v["arg1_range"], 0) == -10000000000000000.0, False))
)

def rule_59_func(arg1, arg2, solver=None, neg=False):
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
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_ndim == arg2.ndim)
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 59
        rule_59(solver, {'arg1_range': arg1_range, 'arg2_range': arg2_range, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_59(solver, {'arg1_range': arg1['range'], 'arg2_range': arg2['range'], 'arg2_ndim': arg2['ndim']}, neg)
