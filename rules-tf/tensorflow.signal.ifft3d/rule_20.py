import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The size of the last dimension should be power of 2 or power of 2 plus 1 for better performance, up to 1024. Expressed using boolean ORs (Rule 20)

rule_20 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 1, Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 2), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 3), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 4), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 5), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 8), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 9), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 16), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 17), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 32), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 33), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 64), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 65), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 128), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 129), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 256), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 257), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 512), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 513), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 1024), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 1025)) if n else
          Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 1, Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 2), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 3), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 4), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 5), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 8), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 9), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 16), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 17), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 32), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 33), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 64), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 65), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 128), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 129), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 256), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 257), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 512), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 513), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 1024), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 1025))
)

def rule_20_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 20
        rule_20(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_20(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape']}, neg)
