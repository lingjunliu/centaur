import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if padding is "valid", input spatial dimensions must be at least the kernel size (Rule 10)

rule_10 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_ndim"] == 4, (If(v["arg3_value"] == 21, (If(v["arg4_value"] == 24, And(Select(v["arg1_shape"], 1) >= v["arg2_value"], Select(v["arg1_shape"], 2) >= v["arg2_value"]), And(Select(v["arg1_shape"], 2) >= v["arg2_value"], Select(v["arg1_shape"], 3) >= v["arg2_value"]))), True)))) if n else
          And(v["arg1_ndim"] == 4, (If(v["arg3_value"] == 21, (If(v["arg4_value"] == 24, And(Select(v["arg1_shape"], 1) >= v["arg2_value"], Select(v["arg1_shape"], 2) >= v["arg2_value"]), And(Select(v["arg1_shape"], 2) >= v["arg2_value"], Select(v["arg1_shape"], 3) >= v["arg2_value"]))), True))))
)

def rule_10_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, str):
            return False
        if not isinstance(arg4, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == list_of_string_values_tf.index(arg3))
        solver.add(arg4_value == list_of_string_values_tf.index(arg4))

        # Constraints for rule 10
        rule_10(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_10(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
