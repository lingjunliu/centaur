import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Householder operator flags: is_positive_definite must be false and is_square must be true. Reflection axis must be a valid tensor, and name is a valid string (Rule 42)

rule_42 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(v["arg1_value"] == False, v["arg2_value"] == True), (Or(Or(Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8), v["arg3_dtype"] == 9), v["arg3_dtype"] == 10))), v["arg3_ndim"] == 1), Select(v["arg3_shape"], 0) > 0), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg4_value"] == 6, v["arg4_value"] == 7), v["arg4_value"] == 8), v["arg4_value"] == 9), v["arg4_value"] == 10), v["arg4_value"] == 11), v["arg4_value"] == 12), v["arg4_value"] == 13), v["arg4_value"] == 14), v["arg4_value"] == 15), v["arg4_value"] == 16), v["arg4_value"] == 17), v["arg4_value"] == 18), v["arg4_value"] == 19), v["arg4_value"] == 20), v["arg4_value"] == 21), v["arg4_value"] == 22), v["arg4_value"] == 23), v["arg4_value"] == 24), v["arg4_value"] == 25)))) if n else
          And(And(And(And(And(v["arg1_value"] == False, v["arg2_value"] == True), (Or(Or(Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8), v["arg3_dtype"] == 9), v["arg3_dtype"] == 10))), v["arg3_ndim"] == 1), Select(v["arg3_shape"], 0) > 0), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg4_value"] == 6, v["arg4_value"] == 7), v["arg4_value"] == 8), v["arg4_value"] == 9), v["arg4_value"] == 10), v["arg4_value"] == 11), v["arg4_value"] == 12), v["arg4_value"] == 13), v["arg4_value"] == 14), v["arg4_value"] == 15), v["arg4_value"] == 16), v["arg4_value"] == 17), v["arg4_value"] == 18), v["arg4_value"] == 19), v["arg4_value"] == 20), v["arg4_value"] == 21), v["arg4_value"] == 22), v["arg4_value"] == 23), v["arg4_value"] == 24), v["arg4_value"] == 25))))
)

def rule_42_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        if not isinstance(arg4, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')
        arg4_value = String('arg4_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_value == list_of_string_values_tf.index(arg4))

        # Constraints for rule 42
        rule_42(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape, 'arg3_ndim': arg3_ndim, 'arg3_dtype': arg3_dtype, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_42(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape'], 'arg3_ndim': arg3['ndim'], 'arg3_dtype': arg3['dtype'], 'arg4_value': arg4['value']}, neg)
