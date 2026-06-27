import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The inputs tensor must be 3D or 4D with 3 channels in either channels_last or channels_first format (Rule 38)

rule_38 = lambda s, v, n=False: (
    s.add(Not(And((Or(v["arg1_ndim"] == 3, v["arg1_ndim"] == 4)), (Or(Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 3, Select(v["arg1_shape"], v["arg1_ndim"] - 3) == 3)))) if n else
          And((Or(v["arg1_ndim"] == 3, v["arg1_ndim"] == 4)), (Or(Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 3, Select(v["arg1_shape"], v["arg1_ndim"] - 3) == 3))))
)

def rule_38_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])

        # Constraints for rule 38
        rule_38(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_38(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape']}, neg)
