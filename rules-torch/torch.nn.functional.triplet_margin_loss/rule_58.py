import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If reduction is not constant and all tensors have more than 0 dimensions, then the margin must be positive and P must be positive. (Rule 58)

rule_58 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And(v["arg1_value"] != 20, v["arg2_ndim"] > 0), v["arg3_ndim"] > 0), v["arg4_ndim"] > 0), And(v["arg5_value"] > 0, v["arg6_value"] > 0), False)) if n else
          If(And(And(And(v["arg1_value"] != 20, v["arg2_ndim"] > 0), v["arg3_ndim"] > 0), v["arg4_ndim"] > 0), And(v["arg5_value"] > 0, v["arg6_value"] > 0), False))
)

def rule_58_func(arg1, arg2, arg3, arg4, arg5, arg6, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False
        if not isinstance(arg5, (float, np.floating)):
            return False
        if not (isinstance(arg6, (int, np.integer)) and not isinstance(arg6, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg3_ndim = Int('arg3_ndim')
        arg4_ndim = Int('arg4_ndim')
        arg5_value = Real('arg5_value')
        arg6_value = Int('arg6_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_torch.index(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg4_ndim == arg4.ndim)
        solver.add(arg5_value == arg5)
        solver.add(arg6_value == int(arg6))

        # Constraints for rule 58
        rule_58(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg3_ndim': arg3_ndim, 'arg4_ndim': arg4_ndim, 'arg5_value': arg5_value, 'arg6_value': arg6_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_58(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg3_ndim': arg3['ndim'], 'arg4_ndim': arg4['ndim'], 'arg5_value': arg5['value'], 'arg6_value': arg6['value']}, neg)
