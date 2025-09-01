import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# stride must be less than or equal to input size plus twice the padding, tuple input (Rule 29)

rule_29 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg1_length"] == 2, v["arg3_length"] == 2), Select(v["arg1_values"], 0) <= Select(v["arg2_shape"], v["arg2_ndim"] - 2) + 2 * Select(v["arg3_values"], 0)), Select(v["arg1_values"], 1) <= Select(v["arg2_shape"], v["arg2_ndim"] - 1) + 2 * Select(v["arg3_values"], 1))) if n else
          And(And(And(v["arg1_length"] == 2, v["arg3_length"] == 2), Select(v["arg1_values"], 0) <= Select(v["arg2_shape"], v["arg2_ndim"] - 2) + 2 * Select(v["arg3_values"], 0)), Select(v["arg1_values"], 1) <= Select(v["arg2_shape"], v["arg2_ndim"] - 1) + 2 * Select(v["arg3_values"], 1)))
)

def rule_29_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 29
        rule_29(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim, 'arg3_length': arg3_length, 'arg3_values': arg3_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_29(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim'], 'arg3_length': arg3['length'], 'arg3_values': arg3['values']}, neg)
