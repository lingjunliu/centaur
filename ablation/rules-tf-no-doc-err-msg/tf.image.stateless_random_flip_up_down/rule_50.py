import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The image must be a tensor with at least 3 dimensions and the height and width must be greater than 0, if the image is 3 dimensional, the last dim must be 1 or 3 (Rule 50)

rule_50 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg1_ndim"] >= 3, Select(v["arg1_shape"], 0) > 0), Select(v["arg1_shape"], 1) > 0), If(v["arg1_ndim"] == 3, (Or(Select(v["arg1_shape"], 2) == 1, Select(v["arg1_shape"], 2) == 3)), True))) if n else
          And(And(And(v["arg1_ndim"] >= 3, Select(v["arg1_shape"], 0) > 0), Select(v["arg1_shape"], 1) > 0), If(v["arg1_ndim"] == 3, (Or(Select(v["arg1_shape"], 2) == 1, Select(v["arg1_shape"], 2) == 3)), True)))
)

def rule_50_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 50
        rule_50(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_50(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape']}, neg)
