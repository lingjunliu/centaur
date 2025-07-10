import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Input tensor must be of floating point, and not complex or boolean, and must have a shape and must have proper range of values. (Rule 80)

rule_80 = lambda s, v, n=False: (
    s.add(Not(And(And((Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 9)), (Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) >= 1) for i in range(6)]))), (If(v["arg1_dtype"] == 7, And(Select(v["arg1_range"], 0) > -6.5e4, Select(v["arg1_range"], 1) < 6.5e4), If(v["arg1_dtype"] == 8, And(Select(v["arg1_range"], 0) > -3.4e38, Select(v["arg1_range"], 1) < 3.4e38), And(Select(v["arg1_range"], 0) > -1.7e308, Select(v["arg1_range"], 1) < 1.7e308)))))) if n else
          And(And((Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 9)), (Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) >= 1) for i in range(6)]))), (If(v["arg1_dtype"] == 7, And(Select(v["arg1_range"], 0) > -6.5e4, Select(v["arg1_range"], 1) < 6.5e4), If(v["arg1_dtype"] == 8, And(Select(v["arg1_range"], 0) > -3.4e38, Select(v["arg1_range"], 1) < 3.4e38), And(Select(v["arg1_range"], 0) > -1.7e308, Select(v["arg1_range"], 1) < 1.7e308))))))
)

def rule_80_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 80
        rule_80(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg1_range': arg1_range, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_80(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg1_range': arg1['range'], 'arg1_shape': arg1['shape']}, neg)
