import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If dilation is a tuple, H_out and W_out must both be positive using tuple dilation values, and 4D input (Rule 85)

rule_85 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 4, And((Select(v["arg1_shape"], 2) - 1) * Select(v["arg2_values"], 0) - 2 * v["arg3_value"] + v["arg4_value"] + 1 > 0, (Select(v["arg1_shape"], 3) - 1) * Select(v["arg2_values"], 1) - 2 * v["arg3_value"] + v["arg4_value"] + 1 > 0), True)) if n else
          If(v["arg1_ndim"] == 4, And((Select(v["arg1_shape"], 2) - 1) * Select(v["arg2_values"], 0) - 2 * v["arg3_value"] + v["arg4_value"] + 1 > 0, (Select(v["arg1_shape"], 3) - 1) * Select(v["arg2_values"], 1) - 2 * v["arg3_value"] + v["arg4_value"] + 1 > 0), True))
)

def rule_85_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == int(arg4))

        # Constraints for rule 85
        rule_85(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_values': arg2_values, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_85(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_values': arg2['values'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
