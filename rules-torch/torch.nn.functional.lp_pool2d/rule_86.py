import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the kernel size is larger than the input after padding the output size will be 0, so no gradient (Rule 86)

rule_86 = lambda s, v, n=False: (
    s.add(Not(If(Or(Select(v["arg2_values"], 0) > Select(v["arg1_shape"], 2) + 2 * Select(v["arg3_values"], 0), Select(v["arg2_values"], 1) > Select(v["arg1_shape"], 3) + 2 * Select(v["arg3_values"], 1)), Select(v["arg1_range"], 0) >= 0, True)) if n else
          If(Or(Select(v["arg2_values"], 0) > Select(v["arg1_shape"], 2) + 2 * Select(v["arg3_values"], 0), Select(v["arg2_values"], 1) > Select(v["arg1_shape"], 3) + 2 * Select(v["arg3_values"], 1)), Select(v["arg1_range"], 0) >= 0, True))
)

def rule_86_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_values = Array('arg3_values', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 86
        rule_86(solver, {'arg1_shape': arg1_shape, 'arg1_range': arg1_range, 'arg2_values': arg2_values, 'arg3_values': arg3_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_86(solver, {'arg1_shape': arg1['shape'], 'arg1_range': arg1['range'], 'arg2_values': arg2['values'], 'arg3_values': arg3['values']}, neg)
