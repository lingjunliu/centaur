import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# inputs shape and data_format must be compatible and have valid spatial dimensions (Rule 21)

rule_21 = lambda s, v, n=False: (
    s.add(Not(And(And((Or(v["arg2_value"] == 24, v["arg2_value"] == 25)), (Or(v["arg1_ndim"] == 3, v["arg1_ndim"] == 4))), (If(v["arg2_value"] == 24, And(Select(v["arg1_shape"], v["arg1_ndim"] - 3) > 0, Select(v["arg1_shape"], v["arg1_ndim"] - 2) > 0), And(Select(v["arg1_shape"], v["arg1_ndim"] - 2) > 0, Select(v["arg1_shape"], v["arg1_ndim"] - 1) > 0))))) if n else
          And(And((Or(v["arg2_value"] == 24, v["arg2_value"] == 25)), (Or(v["arg1_ndim"] == 3, v["arg1_ndim"] == 4))), (If(v["arg2_value"] == 24, And(Select(v["arg1_shape"], v["arg1_ndim"] - 3) > 0, Select(v["arg1_shape"], v["arg1_ndim"] - 2) > 0), And(Select(v["arg1_shape"], v["arg1_ndim"] - 2) > 0, Select(v["arg1_shape"], v["arg1_ndim"] - 1) > 0)))))
)

def rule_21_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))

        # Constraints for rule 21
        rule_21(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_21(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)
