import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If count_include_pad is False, then if padding is a tuple and not all elements zero, then input dimensions must be larger than kernel size + padding * 2 (Rule 67)

rule_67 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == False, (Or(Select(v["arg2_values"], 0) > 0, Select(v["arg2_values"], 1) > 0))), And(And(v["arg4_ndim"] == 4, Select(v["arg4_shape"], 2) > Select(v["arg3_values"], 0) + Select(v["arg2_values"], 0) * 2), Select(v["arg4_shape"], 3) > Select(v["arg3_values"], 1) + Select(v["arg2_values"], 1) * 2), True)) if n else
          If(And(v["arg1_value"] == False, (Or(Select(v["arg2_values"], 0) > 0, Select(v["arg2_values"], 1) > 0))), And(And(v["arg4_ndim"] == 4, Select(v["arg4_shape"], 2) > Select(v["arg3_values"], 0) + Select(v["arg2_values"], 0) * 2), Select(v["arg4_shape"], 3) > Select(v["arg3_values"], 1) + Select(v["arg2_values"], 1) * 2), True))
)

def rule_67_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_values = Array('arg3_values', IntSort(), IntSort())
        arg4_ndim = Int('arg4_ndim')
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])
        solver.add(arg4_ndim == arg4.ndim)
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])

        # Constraints for rule 67
        rule_67(solver, {'arg1_value': arg1_value, 'arg2_values': arg2_values, 'arg3_values': arg3_values, 'arg4_ndim': arg4_ndim, 'arg4_shape': arg4_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_67(solver, {'arg1_value': arg1['value'], 'arg2_values': arg2['values'], 'arg3_values': arg3['values'], 'arg4_ndim': arg4['ndim'], 'arg4_shape': arg4['shape']}, neg)
