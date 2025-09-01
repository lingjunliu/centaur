import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# x must be a tensor with half or float32 or float64 or bfloat16 dtype and its rank must be greater or equal to 0 (Rule 22)

rule_22 = lambda s, v, n=False: (
    s.add(Not(And((Or(Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 12)), v["arg1_ndim"] >= 0)) if n else
          And((Or(Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 12)), v["arg1_ndim"] >= 0))
)

def rule_22_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 22
        rule_22(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_22(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim']}, neg)
