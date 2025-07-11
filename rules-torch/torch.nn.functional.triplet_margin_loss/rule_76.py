import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Input tensors must have at least one dimension, p non-negative and reduction not constant (Rule 76)

rule_76 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(v["arg1_ndim"] >= 1, v["arg2_ndim"] >= 1), v["arg3_ndim"] >= 1), v["arg4_value"] >= 0), v["arg5_value"] != 20)) if n else
          And(And(And(And(v["arg1_ndim"] >= 1, v["arg2_ndim"] >= 1), v["arg3_ndim"] >= 1), v["arg4_value"] >= 0), v["arg5_value"] != 20))
)

def rule_76_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
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
        if not isinstance(arg3, np.ndarray):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False
        if not isinstance(arg5, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_ndim = Int('arg2_ndim')
        arg3_ndim = Int('arg3_ndim')
        arg4_value = Int('arg4_value')
        arg5_value = String('arg5_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg4_value == int(arg4))
        solver.add(arg5_value == list_of_string_values_torch.index(arg5))

        # Constraints for rule 76
        rule_76(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg3_ndim': arg3_ndim, 'arg4_value': arg4_value, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_76(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg3_ndim': arg3['ndim'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value']}, neg)
