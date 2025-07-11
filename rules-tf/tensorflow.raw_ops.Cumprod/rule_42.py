import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If axis is a tensor, it must be a scalar tensor of int32 or int64 dtype AND the axis value has to be within the valid range of x's dimensions (Rule 42)

rule_42 = lambda s, v, n=False: (
    s.add(Not(And(And(And((Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4)), v["arg2_ndim"] == 0), Select(v["arg2_range"], 0) >= (0 - v["arg1_ndim"])), Select(v["arg2_range"], 1) < v["arg1_ndim"])) if n else
          And(And(And((Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4)), v["arg2_ndim"] == 0), Select(v["arg2_range"], 0) >= (0 - v["arg1_ndim"])), Select(v["arg2_range"], 1) < v["arg1_ndim"]))
)

def rule_42_func(arg1, arg2, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 42
        rule_42(solver, {'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype, 'arg2_range': arg2_range, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_42(solver, {'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_range': arg2['range'], 'arg2_ndim': arg2['ndim']}, neg)
