import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Check shape matches, dimensions are within bounds and some string value should be same, and must be > 1 and dtype within range. min value < 0 and ndim should be <5 (Rule 85)

rule_85 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0), v["arg1_ndim"] > 1), v["arg3_value"] == 22), (And(v["arg1_dtype"] > 0, v["arg1_dtype"] < 13))), Select(v["arg1_range"], 0) < 0), v["arg1_ndim"] < 5)) if n else
          And(And(And(And(And(Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0), v["arg1_ndim"] > 1), v["arg3_value"] == 22), (And(v["arg1_dtype"] > 0, v["arg1_dtype"] < 13))), Select(v["arg1_range"], 0) < 0), v["arg1_ndim"] < 5))
)

def rule_85_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = String('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == list_of_string_values_tf.index(arg3))

        # Constraints for rule 85
        rule_85(solver, {'arg1_shape': arg1_shape, 'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_85(solver, {'arg1_shape': arg1['shape'], 'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value']}, neg)
