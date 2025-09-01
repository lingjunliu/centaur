import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Handling shape, kernel_size and padding to prevent Runtime Error, tuple values (Rule 158)

rule_158 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(Select(v["arg1_shape"], v["arg1_ndim"] - 2) + 2 * Select(v["arg3_values"], 0) >= Select(v["arg2_values"], 0), Select(v["arg1_shape"], v["arg1_ndim"] - 1) + 2 * Select(v["arg3_values"], 1) >= Select(v["arg2_values"], 1)), Select(v["arg2_values"], 0) > 0), Select(v["arg2_values"], 1) > 0), Select(v["arg3_values"], 0) >= 0), Select(v["arg3_values"], 1) >= 0)) if n else
          And(And(And(And(And(Select(v["arg1_shape"], v["arg1_ndim"] - 2) + 2 * Select(v["arg3_values"], 0) >= Select(v["arg2_values"], 0), Select(v["arg1_shape"], v["arg1_ndim"] - 1) + 2 * Select(v["arg3_values"], 1) >= Select(v["arg2_values"], 1)), Select(v["arg2_values"], 0) > 0), Select(v["arg2_values"], 1) > 0), Select(v["arg3_values"], 0) >= 0), Select(v["arg3_values"], 1) >= 0))
)

def rule_158_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_values = Array('arg3_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 158
        rule_158(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_values': arg2_values, 'arg3_values': arg3_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_158(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_values': arg2['values'], 'arg3_values': arg3['values']}, neg)
