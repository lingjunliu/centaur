import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if tensor min value is greater than -1 and tensor dtype is not string, then max value must be greater than 1, if there are at least 2 dimensions whose shapes are greater than 1 (Rule 255)

rule_255 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And(And(Select(v["arg1_range"], 0) > -1, v["arg1_dtype"] != 11), (v["arg1_ndim"] > 1)), (Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) > 1) for i in range(6)]))), (Or([And(j < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], j) > 1) for j in range(6)]))), Select(v["arg1_range"], 1) > 1, False)) if n else
          If(And(And(And(And(Select(v["arg1_range"], 0) > -1, v["arg1_dtype"] != 11), (v["arg1_ndim"] > 1)), (Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) > 1) for i in range(6)]))), (Or([And(j < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], j) > 1) for j in range(6)]))), Select(v["arg1_range"], 1) > 1, False))
)

def rule_255_func(arg1, solver=None, neg=False):
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
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 255
        rule_255(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_255(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range'], 'arg1_shape': arg1['shape']}, neg)
