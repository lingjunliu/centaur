import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Replicate Mode implies Value constraint, combined with 1D check to be very different (Rule 95)

rule_95 = lambda s, v, n=False: (
    s.add(Not(And((If(v["arg1_ndim"] == 1, Or(v["arg2_length"] == 0, v["arg2_length"] == 2), True)), (If(v["arg3_value"] == 23, v["arg4_value"] == 0, True)))) if n else
          And((If(v["arg1_ndim"] == 1, Or(v["arg2_length"] == 0, v["arg2_length"] == 2), True)), (If(v["arg3_value"] == 23, v["arg4_value"] == 0, True))))
)

def rule_95_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        if not isinstance(arg3, str):
            return False
        if not isinstance(arg4, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_length = Int('arg2_length')
        arg3_value = Int('arg3_value')
        arg4_value = Real('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_length == len(arg2))
        solver.add(arg3_value == list_of_string_values_torch.index(arg3))
        solver.add(arg4_value == arg4)

        # Constraints for rule 95
        rule_95(solver, {'arg1_ndim': arg1_ndim, 'arg2_length': arg2_length, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_95(solver, {'arg1_ndim': arg1['ndim'], 'arg2_length': arg2['length'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
