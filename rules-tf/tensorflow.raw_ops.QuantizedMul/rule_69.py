import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the output dtype is provided, the shapes of x, y should broadcast and the max of x and y should be greater than or equal to zero (Rule 69)

rule_69 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(Or(Or(v["arg1_value"] == 1, v["arg1_value"] == 5), v["arg1_value"] == 3), v["arg1_value"] == 2), v["arg1_value"] == 14), And([Implies(i < (If(v["arg2_ndim"] >= v["arg3_ndim"], v["arg2_ndim"] - 1, v["arg3_ndim"] - 1) + 1), And(And(Or(Or(Or(Or(v["arg2_ndim"] - i - 1 < 0, v["arg3_ndim"] - i - 1 < 0), Select(v["arg2_shape"], v["arg2_ndim"] - i - 1) == 1), Select(v["arg3_shape"], v["arg3_ndim"] - i - 1) == 1), Select(v["arg2_shape"], v["arg2_ndim"] - i - 1) == Select(v["arg3_shape"], v["arg3_ndim"] - i - 1)), Select(v["arg2_range"], 1) >= 0), Select(v["arg3_range"], 1) >= 0)) for i in range(6)]), False)) if n else
          If(Or(Or(Or(Or(v["arg1_value"] == 1, v["arg1_value"] == 5), v["arg1_value"] == 3), v["arg1_value"] == 2), v["arg1_value"] == 14), And([Implies(i < (If(v["arg2_ndim"] >= v["arg3_ndim"], v["arg2_ndim"] - 1, v["arg3_ndim"] - 1) + 1), And(And(Or(Or(Or(Or(v["arg2_ndim"] - i - 1 < 0, v["arg3_ndim"] - i - 1 < 0), Select(v["arg2_shape"], v["arg2_ndim"] - i - 1) == 1), Select(v["arg3_shape"], v["arg3_ndim"] - i - 1) == 1), Select(v["arg2_shape"], v["arg2_ndim"] - i - 1) == Select(v["arg3_shape"], v["arg3_ndim"] - i - 1)), Select(v["arg2_range"], 1) >= 0), Select(v["arg3_range"], 1) >= 0)) for i in range(6)]), False))
)

def rule_69_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_range = Array('arg3_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 69
        rule_69(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg2_range': arg2_range, 'arg2_ndim': arg2_ndim, 'arg3_shape': arg3_shape, 'arg3_range': arg3_range, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_69(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg2_range': arg2['range'], 'arg2_ndim': arg2['ndim'], 'arg3_shape': arg3['shape'], 'arg3_range': arg3['range'], 'arg3_ndim': arg3['ndim']}, neg)
