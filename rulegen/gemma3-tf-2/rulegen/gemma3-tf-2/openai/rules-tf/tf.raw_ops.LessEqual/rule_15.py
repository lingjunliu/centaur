import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# well-formed LessEqual call: supported dtype, same dtype, broadcastable shapes, and valid name (Rule 15)

rule_15 = lambda s, v, n=False: (
    s.add(Not(And(And(And((Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 5), v["arg1_dtype"] == 6), v["arg1_dtype"] == 7), v["arg1_dtype"] == 8)), v["arg1_dtype"] == v["arg2_dtype"]), (And([Implies(i < (If(v["arg1_ndim"] >= v["arg2_ndim"], v["arg1_ndim"] - 1, v["arg2_ndim"] - 1) + 1), Or(Or(Or(Or(v["arg1_ndim"] - i - 1 < 0, v["arg2_ndim"] - i - 1 < 0), Select(v["arg1_shape"], v["arg1_ndim"] - i - 1) == 1), Select(v["arg2_shape"], v["arg2_ndim"] - i - 1) == 1), Select(v["arg1_shape"], v["arg1_ndim"] - i - 1) == Select(v["arg2_shape"], v["arg2_ndim"] - i - 1))) for i in range(6)]))), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg3_value"] == 0, v["arg3_value"] == 1), v["arg3_value"] == 2), v["arg3_value"] == 3), v["arg3_value"] == 4), v["arg3_value"] == 5), v["arg3_value"] == 6), v["arg3_value"] == 7), v["arg3_value"] == 8), v["arg3_value"] == 9), v["arg3_value"] == 10), v["arg3_value"] == 11), v["arg3_value"] == 12), v["arg3_value"] == 13), v["arg3_value"] == 14), v["arg3_value"] == 15), v["arg3_value"] == 16), v["arg3_value"] == 17), v["arg3_value"] == 18), v["arg3_value"] == 19), v["arg3_value"] == 20), v["arg3_value"] == 21), v["arg3_value"] == 22), v["arg3_value"] == 23), v["arg3_value"] == 24), v["arg3_value"] == 25)))) if n else
          And(And(And((Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 5), v["arg1_dtype"] == 6), v["arg1_dtype"] == 7), v["arg1_dtype"] == 8)), v["arg1_dtype"] == v["arg2_dtype"]), (And([Implies(i < (If(v["arg1_ndim"] >= v["arg2_ndim"], v["arg1_ndim"] - 1, v["arg2_ndim"] - 1) + 1), Or(Or(Or(Or(v["arg1_ndim"] - i - 1 < 0, v["arg2_ndim"] - i - 1 < 0), Select(v["arg1_shape"], v["arg1_ndim"] - i - 1) == 1), Select(v["arg2_shape"], v["arg2_ndim"] - i - 1) == 1), Select(v["arg1_shape"], v["arg1_ndim"] - i - 1) == Select(v["arg2_shape"], v["arg2_ndim"] - i - 1))) for i in range(6)]))), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg3_value"] == 0, v["arg3_value"] == 1), v["arg3_value"] == 2), v["arg3_value"] == 3), v["arg3_value"] == 4), v["arg3_value"] == 5), v["arg3_value"] == 6), v["arg3_value"] == 7), v["arg3_value"] == 8), v["arg3_value"] == 9), v["arg3_value"] == 10), v["arg3_value"] == 11), v["arg3_value"] == 12), v["arg3_value"] == 13), v["arg3_value"] == 14), v["arg3_value"] == 15), v["arg3_value"] == 16), v["arg3_value"] == 17), v["arg3_value"] == 18), v["arg3_value"] == 19), v["arg3_value"] == 20), v["arg3_value"] == 21), v["arg3_value"] == 22), v["arg3_value"] == 23), v["arg3_value"] == 24), v["arg3_value"] == 25))))
)

def rule_15_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg3_value = String('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_value == list_of_string_values_tf.index(arg3))

        # Constraints for rule 15
        rule_15(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_15(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value']}, neg)
