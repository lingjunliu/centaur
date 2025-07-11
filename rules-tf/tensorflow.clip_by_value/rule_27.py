import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if clip_value_max is a tensor and clip_value_min is a scalar, the dtype of clip_value_max and clip_value_min must be compatible. (Rule 27)

rule_27 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_ndim"] > 0, v["arg1_ndim"] == 0), Or(Or(Or(Or(Or(Or((And(v["arg2_dtype"] == 7, v["arg1_dtype"] == 7)), (And(v["arg2_dtype"] == 8, v["arg1_dtype"] == 8))), (And(v["arg2_dtype"] == 1, v["arg1_dtype"] == 1))), (And(v["arg2_dtype"] == 2, v["arg1_dtype"] == 2))), (And(v["arg2_dtype"] == 3, v["arg1_dtype"] == 3))), (And(v["arg2_dtype"] == 4, v["arg1_dtype"] == 4))), (And(v["arg2_dtype"] == 5, v["arg1_dtype"] == 5))), False)) if n else
          If(And(v["arg2_ndim"] > 0, v["arg1_ndim"] == 0), Or(Or(Or(Or(Or(Or((And(v["arg2_dtype"] == 7, v["arg1_dtype"] == 7)), (And(v["arg2_dtype"] == 8, v["arg1_dtype"] == 8))), (And(v["arg2_dtype"] == 1, v["arg1_dtype"] == 1))), (And(v["arg2_dtype"] == 2, v["arg1_dtype"] == 2))), (And(v["arg2_dtype"] == 3, v["arg1_dtype"] == 3))), (And(v["arg2_dtype"] == 4, v["arg1_dtype"] == 4))), (And(v["arg2_dtype"] == 5, v["arg1_dtype"] == 5))), False))
)

def rule_27_func(arg1, arg2, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 27
        rule_27(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_27(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim']}, neg)
