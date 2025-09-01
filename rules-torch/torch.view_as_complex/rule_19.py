import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The input tensor must have either float64, float32, or float16 dtype, and its last dimension must have size 2 (Rule 19)

rule_19 = lambda s, v, n=False: (
    s.add(Not(And((Or(Or(v["arg1_dtype"] == 8, v["arg1_dtype"] == 7), v["arg1_dtype"] == 6)), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 2)) if n else
          And((Or(Or(v["arg1_dtype"] == 8, v["arg1_dtype"] == 7), v["arg1_dtype"] == 6)), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 2))
)

def rule_19_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 19
        rule_19(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_19(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim']}, neg)
