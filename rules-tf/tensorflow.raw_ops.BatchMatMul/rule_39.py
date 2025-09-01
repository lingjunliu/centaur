import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The inner-most dimensions must be compatible for matrix multiplication based on adj_x and adj_y flags, uses only the flag variables (Rule 39)

rule_39 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == True, v["arg2_value"] == True), Select(v["arg3_shape"], v["arg3_ndim"] - 2) == Select(v["arg4_shape"], v["arg4_ndim"] - 1), If(And(v["arg1_value"] == True, v["arg2_value"] == False), Select(v["arg3_shape"], v["arg3_ndim"] - 2) == Select(v["arg4_shape"], v["arg4_ndim"] - 2), If(And(v["arg1_value"] == False, v["arg2_value"] == True), Select(v["arg3_shape"], v["arg3_ndim"] - 1) == Select(v["arg4_shape"], v["arg4_ndim"] - 1), Select(v["arg3_shape"], v["arg3_ndim"] - 1) == Select(v["arg4_shape"], v["arg4_ndim"] - 2))))) if n else
          If(And(v["arg1_value"] == True, v["arg2_value"] == True), Select(v["arg3_shape"], v["arg3_ndim"] - 2) == Select(v["arg4_shape"], v["arg4_ndim"] - 1), If(And(v["arg1_value"] == True, v["arg2_value"] == False), Select(v["arg3_shape"], v["arg3_ndim"] - 2) == Select(v["arg4_shape"], v["arg4_ndim"] - 2), If(And(v["arg1_value"] == False, v["arg2_value"] == True), Select(v["arg3_shape"], v["arg3_ndim"] - 1) == Select(v["arg4_shape"], v["arg4_ndim"] - 1), Select(v["arg3_shape"], v["arg3_ndim"] - 1) == Select(v["arg4_shape"], v["arg4_ndim"] - 2)))))
)

def rule_39_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_ndim = Int('arg4_ndim')
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg4_ndim == arg4.ndim)
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])

        # Constraints for rule 39
        rule_39(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape, 'arg3_ndim': arg3_ndim, 'arg4_shape': arg4_shape, 'arg4_ndim': arg4_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_39(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape'], 'arg3_ndim': arg3['ndim'], 'arg4_shape': arg4['shape'], 'arg4_ndim': arg4['ndim']}, neg)
