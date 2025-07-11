import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# If s is given and dim is not given, then the last dimension size should be power of 2 plus one (Rule 32)

rule_32 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 3, Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 5), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 9), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 17), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 33), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 65), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 129), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 257), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 513)) if n else
          Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 3, Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 5), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 9), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 17), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 33), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 65), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 129), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 257), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 513))
)

def rule_32_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 32
        rule_32(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_32(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim']}, neg)
