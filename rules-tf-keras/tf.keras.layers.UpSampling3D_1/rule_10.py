import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# upsampled spatial dimensions must not exceed memory limit of 2048 (Rule 10)

rule_10 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg3_ndim"] == 5, v["arg1_length"] == 3), (If(v["arg2_value"] == 24, (And([Implies(i < (2 + 1), Select(v["arg3_shape"], i + 1) * Select(v["arg1_values"], i) <= 2048) for i in range(6)])), (And([Implies(i < (2 + 1), Select(v["arg3_shape"], i + 2) * Select(v["arg1_values"], i) <= 2048) for i in range(6)])))))) if n else
          And(And(v["arg3_ndim"] == 5, v["arg1_length"] == 3), (If(v["arg2_value"] == 24, (And([Implies(i < (2 + 1), Select(v["arg3_shape"], i + 1) * Select(v["arg1_values"], i) <= 2048) for i in range(6)])), (And([Implies(i < (2 + 1), Select(v["arg3_shape"], i + 2) * Select(v["arg1_values"], i) <= 2048) for i in range(6)]))))))
)

def rule_10_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, str):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_value = String('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 10
        rule_10(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length, 'arg2_value': arg2_value, 'arg3_ndim': arg3_ndim, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_10(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length'], 'arg2_value': arg2['value'], 'arg3_ndim': arg3['ndim'], 'arg3_shape': arg3['shape']}, neg)
