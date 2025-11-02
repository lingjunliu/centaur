import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the tensor is 3d then its dimensions shouldn't exceed 512 (Rule 79)

rule_79 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 3, And(And(Select(v["arg1_shape"], 0) < 512, Select(v["arg1_shape"], 1) < 512), Select(v["arg1_shape"], 2) < 512), True)) if n else
          If(v["arg1_ndim"] == 3, And(And(Select(v["arg1_shape"], 0) < 512, Select(v["arg1_shape"], 1) < 512), Select(v["arg1_shape"], 2) < 512), True))
)

def rule_79_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 79
        rule_79(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_79(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape']}, neg)
