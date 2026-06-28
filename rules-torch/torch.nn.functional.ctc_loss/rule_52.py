import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If targets is concatenated, reduction is not none and zero_infinity is false, target_lengths must be greater than 0 (Rule 52)

rule_52 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_ndim"] == 1, v["arg3_value"] != 6), v["arg4_value"] == False), Select(v["arg2_range"], 1) > 0, True)) if n else
          If(And(And(v["arg1_ndim"] == 1, v["arg3_value"] != 6), v["arg4_value"] == False), Select(v["arg2_range"], 1) > 0, True))
)

def rule_52_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, str):
            return False
        if not isinstance(arg4, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_value = Int('arg3_value')
        arg4_value = Bool('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_value == list_of_string_values_torch.index(arg3))
        solver.add(arg4_value == arg4)

        # Constraints for rule 52
        rule_52(solver, {'arg1_ndim': arg1_ndim, 'arg2_range': arg2_range, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_52(solver, {'arg1_ndim': arg1['ndim'], 'arg2_range': arg2['range'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
