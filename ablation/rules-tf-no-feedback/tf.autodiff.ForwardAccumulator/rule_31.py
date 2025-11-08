import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If primals is an int8, int16, int32, int64, uint8, float16, float32, float64, complex64, complex128 or bool tensor, tangents should also be the same numerical type or scalar 0 or bool tensor  (Rule 31)

rule_31 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or((And(v["arg1_dtype"] == 0, (Or(v["arg2_dtype"] == 0, v["arg2_ndim"] == 0)))), (And(v["arg1_dtype"] == 1, (Or(v["arg2_dtype"] == 1, v["arg2_ndim"] == 0))))), (And(v["arg1_dtype"] == 2, (Or(v["arg2_dtype"] == 2, v["arg2_ndim"] == 0))))), (And(v["arg1_dtype"] == 3, (Or(v["arg2_dtype"] == 3, v["arg2_ndim"] == 0))))), (And(v["arg1_dtype"] == 4, (Or(v["arg2_dtype"] == 4, v["arg2_ndim"] == 0))))), (And(v["arg1_dtype"] == 5, (Or(v["arg2_dtype"] == 5, v["arg2_ndim"] == 0))))), (And(v["arg1_dtype"] == 6, (Or(v["arg2_dtype"] == 6, v["arg2_ndim"] == 0))))), (And(v["arg1_dtype"] == 7, (Or(v["arg2_dtype"] == 7, v["arg2_ndim"] == 0))))), (And(v["arg1_dtype"] == 8, (Or(v["arg2_dtype"] == 8, v["arg2_ndim"] == 0))))), (And(v["arg1_dtype"] == 9, (Or(v["arg2_dtype"] == 9, v["arg2_ndim"] == 0))))), (And(v["arg1_dtype"] == 10, (Or(v["arg2_dtype"] == 10, v["arg2_ndim"] == 0)))))) if n else
          Or(Or(Or(Or(Or(Or(Or(Or(Or(Or((And(v["arg1_dtype"] == 0, (Or(v["arg2_dtype"] == 0, v["arg2_ndim"] == 0)))), (And(v["arg1_dtype"] == 1, (Or(v["arg2_dtype"] == 1, v["arg2_ndim"] == 0))))), (And(v["arg1_dtype"] == 2, (Or(v["arg2_dtype"] == 2, v["arg2_ndim"] == 0))))), (And(v["arg1_dtype"] == 3, (Or(v["arg2_dtype"] == 3, v["arg2_ndim"] == 0))))), (And(v["arg1_dtype"] == 4, (Or(v["arg2_dtype"] == 4, v["arg2_ndim"] == 0))))), (And(v["arg1_dtype"] == 5, (Or(v["arg2_dtype"] == 5, v["arg2_ndim"] == 0))))), (And(v["arg1_dtype"] == 6, (Or(v["arg2_dtype"] == 6, v["arg2_ndim"] == 0))))), (And(v["arg1_dtype"] == 7, (Or(v["arg2_dtype"] == 7, v["arg2_ndim"] == 0))))), (And(v["arg1_dtype"] == 8, (Or(v["arg2_dtype"] == 8, v["arg2_ndim"] == 0))))), (And(v["arg1_dtype"] == 9, (Or(v["arg2_dtype"] == 9, v["arg2_ndim"] == 0))))), (And(v["arg1_dtype"] == 10, (Or(v["arg2_dtype"] == 10, v["arg2_ndim"] == 0))))))
)

def rule_31_func(arg1, arg2, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 31
        rule_31(solver, {'arg1_dtype': arg1_dtype, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_31(solver, {'arg1_dtype': arg1['dtype'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype']}, neg)
