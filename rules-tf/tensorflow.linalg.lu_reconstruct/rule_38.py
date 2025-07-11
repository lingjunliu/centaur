import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If perm is not "none", and validate_args is true, then the last dimension of perm should be equal to the last dimension of the lower_upper matrix (Rule 38)

rule_38 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] != 6, v["arg4_value"] == True), Select(v["arg2_shape"], v["arg2_ndim"] - 1) == Select(v["arg3_shape"], v["arg3_ndim"] - 1), False)) if n else
          If(And(v["arg1_value"] != 6, v["arg4_value"] == True), Select(v["arg2_shape"], v["arg2_ndim"] - 1) == Select(v["arg3_shape"], v["arg3_ndim"] - 1), False))
)

def rule_38_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_value = Bool('arg4_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg4_value == arg4)

        # Constraints for rule 38
        rule_38(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape, 'arg3_ndim': arg3_ndim, 'arg3_shape': arg3_shape, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_38(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'], 'arg3_ndim': arg3['ndim'], 'arg3_shape': arg3['shape'], 'arg4_value': arg4['value']}, neg)
