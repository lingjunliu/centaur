import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If fft_length is a tensor, it must be a scalar int32, its max must be less than 2000000, the input must be complex, fft_length must be positive, and if the input has dimension the innermost dimension must be greater than 1 or 0 (Rule 41)

rule_41 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(v["arg2_dtype"] == 3, v["arg2_ndim"] == 0), Select(v["arg2_range"], 1) < 2000000), (Or(v["arg1_dtype"] == 10, v["arg1_dtype"] == 11))), Select(v["arg2_range"], 0) > 0), (If(v["arg1_ndim"] > 0, (Or(Select(v["arg1_shape"], v["arg1_ndim"] - 1) > 1, Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 0)), True)))) if n else
          And(And(And(And(And(v["arg2_dtype"] == 3, v["arg2_ndim"] == 0), Select(v["arg2_range"], 1) < 2000000), (Or(v["arg1_dtype"] == 10, v["arg1_dtype"] == 11))), Select(v["arg2_range"], 0) > 0), (If(v["arg1_ndim"] > 0, (Or(Select(v["arg1_shape"], v["arg1_ndim"] - 1) > 1, Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 0)), True))))
)

def rule_41_func(arg1, arg2, solver=None, neg=False):
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
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 41
        rule_41(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype, 'arg2_range': arg2_range, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_41(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_range': arg2['range'], 'arg2_ndim': arg2['ndim']}, neg)
