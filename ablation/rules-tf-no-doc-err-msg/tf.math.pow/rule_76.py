import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the first tensor's shape's first element is greater than 100 and ndim is greater than 0 and dtype is not bool and max is less than 500, then the second tensor's dimension must be less or equal than 2 or it must be zero and dtype must be greater than 1 and less than 10 (Rule 76)

rule_76 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And(Select(v["arg1_shape"], 0) > 100, v["arg1_ndim"] > 0), v["arg1_dtype"] != 0), Select(v["arg1_range"], 1) < 500), And(And((Or(v["arg2_ndim"] <= 2, v["arg2_ndim"] == 0)), v["arg2_dtype"] > 1), v["arg2_dtype"] < 10), True)) if n else
          If(And(And(And(Select(v["arg1_shape"], 0) > 100, v["arg1_ndim"] > 0), v["arg1_dtype"] != 0), Select(v["arg1_range"], 1) < 500), And(And((Or(v["arg2_ndim"] <= 2, v["arg2_ndim"] == 0)), v["arg2_dtype"] > 1), v["arg2_dtype"] < 10), True))
)

def rule_76_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 76
        rule_76(solver, {'arg1_ndim': arg1_ndim, 'arg1_range': arg1_range, 'arg1_shape': arg1_shape, 'arg1_dtype': arg1_dtype, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_76(solver, {'arg1_ndim': arg1['ndim'], 'arg1_range': arg1['range'], 'arg1_shape': arg1['shape'], 'arg1_dtype': arg1['dtype'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype']}, neg)
