import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if the shape is 3x3 then min value must be less than 50 (Rule 68)

rule_68 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_ndim"] == 2, Select(v["arg1_shape"], 0) == 3), Select(v["arg1_shape"], 1) == 3), Select(v["arg1_range"], 0) < 50, False)) if n else
          If(And(And(v["arg1_ndim"] == 2, Select(v["arg1_shape"], 0) == 3), Select(v["arg1_shape"], 1) == 3), Select(v["arg1_range"], 0) < 50, False))
)

def rule_68_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 68
        rule_68(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_68(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape']}, neg)
