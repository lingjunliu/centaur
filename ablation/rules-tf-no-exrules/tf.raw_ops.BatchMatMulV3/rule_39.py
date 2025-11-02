import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# x and y must be tensors with compatible dtypes, Tout must be compatible if specified, tensors must be at least 2D (Rule 39)

rule_39 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And((Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 5), v["arg1_dtype"] == 1), v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10)), (Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg2_dtype"] == 6, v["arg2_dtype"] == 7), v["arg2_dtype"] == 8), v["arg2_dtype"] == 5), v["arg2_dtype"] == 1), v["arg2_dtype"] == 2), v["arg2_dtype"] == 3), v["arg2_dtype"] == 4), v["arg2_dtype"] == 9), v["arg2_dtype"] == 10))), v["arg1_ndim"] >= 2), v["arg2_ndim"] >= 2), (Or(Or(Or(Or(Or(Or(Or(Or(v["arg3_value"] == 0, v["arg3_value"] == 6), v["arg3_value"] == 7), v["arg3_value"] == 8), v["arg3_value"] == 2), v["arg3_value"] == 3), v["arg3_value"] == 4), v["arg3_value"] == 9), v["arg3_value"] == 10))), If(v["arg3_value"] != 0, And(v["arg1_dtype"] == v["arg3_value"], v["arg2_dtype"] == v["arg3_value"]), True))) if n else
          And(And(And(And(And((Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 5), v["arg1_dtype"] == 1), v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10)), (Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg2_dtype"] == 6, v["arg2_dtype"] == 7), v["arg2_dtype"] == 8), v["arg2_dtype"] == 5), v["arg2_dtype"] == 1), v["arg2_dtype"] == 2), v["arg2_dtype"] == 3), v["arg2_dtype"] == 4), v["arg2_dtype"] == 9), v["arg2_dtype"] == 10))), v["arg1_ndim"] >= 2), v["arg2_ndim"] >= 2), (Or(Or(Or(Or(Or(Or(Or(Or(v["arg3_value"] == 0, v["arg3_value"] == 6), v["arg3_value"] == 7), v["arg3_value"] == 8), v["arg3_value"] == 2), v["arg3_value"] == 3), v["arg3_value"] == 4), v["arg3_value"] == 9), v["arg3_value"] == 10))), If(v["arg3_value"] != 0, And(v["arg1_dtype"] == v["arg3_value"], v["arg2_dtype"] == v["arg3_value"]), True)))
)

def rule_39_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, torch.dtype) or isinstance(arg3, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_value == list_of_available_dtypes.index(np_dtype(arg3)))

        # Constraints for rule 39
        rule_39(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_39(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype'], 'arg3_value': arg3['value']}, neg)
