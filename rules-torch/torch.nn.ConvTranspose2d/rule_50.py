import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# output_size must be compatible with other parameters (Rule 50)

rule_50 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_ndim"] == 4, And(Select(v["arg1_values"], 0) == (Select(v["arg2_shape"], 2) - 1) * v["arg4_value"] - 2 * v["arg5_value"] + v["arg6_value"] * (v["arg3_value"] - 1) + v["arg7_value"] + 1, Select(v["arg1_values"], 1) == (Select(v["arg2_shape"], 3) - 1) * v["arg4_value"] - 2 * v["arg5_value"] + v["arg6_value"] * (v["arg3_value"] - 1) + v["arg7_value"] + 1), True)) if n else
          If(v["arg2_ndim"] == 4, And(Select(v["arg1_values"], 0) == (Select(v["arg2_shape"], 2) - 1) * v["arg4_value"] - 2 * v["arg5_value"] + v["arg6_value"] * (v["arg3_value"] - 1) + v["arg7_value"] + 1, Select(v["arg1_values"], 1) == (Select(v["arg2_shape"], 3) - 1) * v["arg4_value"] - 2 * v["arg5_value"] + v["arg6_value"] * (v["arg3_value"] - 1) + v["arg7_value"] + 1), True))
)

def rule_50_func(arg1, arg2, arg3, arg4, arg5, arg6, arg7, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))
    arg7 = next(iter(arg7.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False
        if not (isinstance(arg5, (int, np.integer)) and not isinstance(arg5, bool)):
            return False
        if not (isinstance(arg6, (int, np.integer)) and not isinstance(arg6, bool)):
            return False
        if not (isinstance(arg7, (int, np.integer)) and not isinstance(arg7, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')
        arg5_value = Int('arg5_value')
        arg6_value = Int('arg6_value')
        arg7_value = Int('arg7_value')

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == int(arg4))
        solver.add(arg5_value == int(arg5))
        solver.add(arg6_value == int(arg6))
        solver.add(arg7_value == int(arg7))

        # Constraints for rule 50
        rule_50(solver, {'arg1_values': arg1_values, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value, 'arg6_value': arg6_value, 'arg7_value': arg7_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_50(solver, {'arg1_values': arg1['values'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value'], 'arg6_value': arg6['value'], 'arg7_value': arg7['value']}, neg)
