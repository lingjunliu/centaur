import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The difference between the input size plus twice the padding, and the dilation times (kernel size minus 1 (Rule 43)

rule_43 = lambda s, v, n=False: (
    s.add(Not(And((Select(v["arg1_shape"], 2) + 2 * Select(v["arg4_values"], 0) - (Select(v["arg5_values"], 0) * (Select(v["arg2_shape"], 2) - 1) + 1)) % Select(v["arg3_values"], 0) == 0, (Select(v["arg1_shape"], 3) + 2 * Select(v["arg4_values"], 1) - (Select(v["arg5_values"], 1) * (Select(v["arg2_shape"], 3) - 1) + 1)) % Select(v["arg3_values"], 1) == 0)) if n else
          And((Select(v["arg1_shape"], 2) + 2 * Select(v["arg4_values"], 0) - (Select(v["arg5_values"], 0) * (Select(v["arg2_shape"], 2) - 1) + 1)) % Select(v["arg3_values"], 0) == 0, (Select(v["arg1_shape"], 3) + 2 * Select(v["arg4_values"], 1) - (Select(v["arg5_values"], 1) * (Select(v["arg2_shape"], 3) - 1) + 1)) % Select(v["arg3_values"], 1) == 0))
)

def rule_43_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False
        if not (isinstance(arg4, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False
        if not (isinstance(arg5, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg5)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_values = Array('arg3_values', IntSort(), IntSort())
        arg4_values = Array('arg4_values', IntSort(), IntSort())
        arg5_values = Array('arg5_values', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])
        for i in range(len(arg4)):
            arg4_values = Store(arg4_values, i, arg4[i])
        for i in range(len(arg5)):
            arg5_values = Store(arg5_values, i, arg5[i])

        # Constraints for rule 43
        rule_43(solver, {'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg3_values': arg3_values, 'arg4_values': arg4_values, 'arg5_values': arg5_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_43(solver, {'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg3_values': arg3['values'], 'arg4_values': arg4['values'], 'arg5_values': arg5['values']}, neg)
