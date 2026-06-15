import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# avg_pool2d input spatial size must support window size (Rule 71)

rule_71 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg3_value"] == 26, v["arg1_ndim"] == 4), (If(v["arg4_value"] == 31, (If(v["arg2_length"] == 1, And(Select(v["arg1_shape"], 1) >= Select(v["arg2_values"], 0), Select(v["arg1_shape"], 2) >= Select(v["arg2_values"], 0)), If(v["arg2_length"] == 2, And(Select(v["arg1_shape"], 1) >= Select(v["arg2_values"], 0), Select(v["arg1_shape"], 2) >= Select(v["arg2_values"], 1)), If(v["arg2_length"] == 4, And(Select(v["arg1_shape"], 1) >= Select(v["arg2_values"], 1), Select(v["arg1_shape"], 2) >= Select(v["arg2_values"], 2)), True)))), If(v["arg4_value"] == 32, (If(v["arg2_length"] == 1, And(Select(v["arg1_shape"], 2) >= Select(v["arg2_values"], 0), Select(v["arg1_shape"], 3) >= Select(v["arg2_values"], 0)), If(v["arg2_length"] == 2, And(Select(v["arg1_shape"], 2) >= Select(v["arg2_values"], 0), Select(v["arg1_shape"], 3) >= Select(v["arg2_values"], 1)), If(v["arg2_length"] == 4, And(Select(v["arg1_shape"], 2) >= Select(v["arg2_values"], 2), Select(v["arg1_shape"], 3) >= Select(v["arg2_values"], 3)), True)))), True))), True)) if n else
          If(And(v["arg3_value"] == 26, v["arg1_ndim"] == 4), (If(v["arg4_value"] == 31, (If(v["arg2_length"] == 1, And(Select(v["arg1_shape"], 1) >= Select(v["arg2_values"], 0), Select(v["arg1_shape"], 2) >= Select(v["arg2_values"], 0)), If(v["arg2_length"] == 2, And(Select(v["arg1_shape"], 1) >= Select(v["arg2_values"], 0), Select(v["arg1_shape"], 2) >= Select(v["arg2_values"], 1)), If(v["arg2_length"] == 4, And(Select(v["arg1_shape"], 1) >= Select(v["arg2_values"], 1), Select(v["arg1_shape"], 2) >= Select(v["arg2_values"], 2)), True)))), If(v["arg4_value"] == 32, (If(v["arg2_length"] == 1, And(Select(v["arg1_shape"], 2) >= Select(v["arg2_values"], 0), Select(v["arg1_shape"], 3) >= Select(v["arg2_values"], 0)), If(v["arg2_length"] == 2, And(Select(v["arg1_shape"], 2) >= Select(v["arg2_values"], 0), Select(v["arg1_shape"], 3) >= Select(v["arg2_values"], 1)), If(v["arg2_length"] == 4, And(Select(v["arg1_shape"], 2) >= Select(v["arg2_values"], 2), Select(v["arg1_shape"], 3) >= Select(v["arg2_values"], 3)), True)))), True))), True))
)

def rule_71_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, str):
            return False
        if not isinstance(arg4, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_value == list_of_string_values_tf.index(arg3))
        solver.add(arg4_value == list_of_string_values_tf.index(arg4))

        # Constraints for rule 71
        rule_71(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_values': arg2_values, 'arg2_length': arg2_length, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_71(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_values': arg2['values'], 'arg2_length': arg2['length'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
