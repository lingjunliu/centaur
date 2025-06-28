import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if tensor dimension is 2 and minimum value is smaller than -1000 and maximum value is higher than 1000 then shape product can not be smaller than 20 (Rule 245)

rule_245 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_ndim"] == 2, Select(v["arg1_range"], 0) < -1000), Select(v["arg1_range"], 1) > 1000), Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) >= 20, False)) if n else
          If(And(And(v["arg1_ndim"] == 2, Select(v["arg1_range"], 0) < -1000), Select(v["arg1_range"], 1) > 1000), Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) >= 20, False))
)

def rule_245_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 245
        rule_245(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_245(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape']}, neg)
