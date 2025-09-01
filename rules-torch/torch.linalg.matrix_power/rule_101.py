import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If A has more than one dimension, the last two dimensions must match if n=0 and ndim < 5 AND dtype != 8. And shape(A,0 (Rule 101)

rule_101 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And(v["arg1_ndim"] > 1, v["arg2_value"] == 0), v["arg1_ndim"] < 5), v["arg1_dtype"] != 8), And(And(And(Select(v["arg1_shape"], v["arg1_ndim"] - 1) == Select(v["arg1_shape"], v["arg1_ndim"] - 2), Select(v["arg1_shape"], 0) < 10), Select(v["arg1_shape"], 1) > 2), Select(v["arg1_range"], 1) < 100), True)) if n else
          If(And(And(And(v["arg1_ndim"] > 1, v["arg2_value"] == 0), v["arg1_ndim"] < 5), v["arg1_dtype"] != 8), And(And(And(Select(v["arg1_shape"], v["arg1_ndim"] - 1) == Select(v["arg1_shape"], v["arg1_ndim"] - 2), Select(v["arg1_shape"], 0) < 10), Select(v["arg1_shape"], 1) > 2), Select(v["arg1_range"], 1) < 100), True))
)

def rule_101_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 101
        rule_101(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_101(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)
