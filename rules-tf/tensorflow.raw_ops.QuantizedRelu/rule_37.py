import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If out_type is not specified, features and out_type are compatible, and min/max are scalar, then the shape of activations is the same as features (Rule 37)

rule_37 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And(v["arg4_value"] == 0, (Or(Or(Or(Or((v["arg1_dtype"] == 1), (v["arg1_dtype"] == 5)), (v["arg1_dtype"] == 3)), (v["arg1_dtype"] == 2)), (v["arg1_dtype"] == 14)))), v["arg2_ndim"] == 0), v["arg3_ndim"] == 0), True, False)) if n else
          If(And(And(And(v["arg4_value"] == 0, (Or(Or(Or(Or((v["arg1_dtype"] == 1), (v["arg1_dtype"] == 5)), (v["arg1_dtype"] == 3)), (v["arg1_dtype"] == 2)), (v["arg1_dtype"] == 14)))), v["arg2_ndim"] == 0), v["arg3_ndim"] == 0), True, False))
)

def rule_37_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not (isinstance(arg4, torch.dtype) or isinstance(arg4, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg3_ndim = Int('arg3_ndim')
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg4_value == list_of_available_dtypes.index(np_dtype(arg4)))

        # Constraints for rule 37
        rule_37(solver, {'arg1_dtype': arg1_dtype, 'arg2_ndim': arg2_ndim, 'arg3_ndim': arg3_ndim, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_37(solver, {'arg1_dtype': arg1['dtype'], 'arg2_ndim': arg2['ndim'], 'arg3_ndim': arg3['ndim'], 'arg4_value': arg4['value']}, neg)
