import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Input is Tensor, kernel_size as int, stride as int, padding as int, dilation as tuple(int (Rule 59)

rule_59 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And((Or(v["arg1_ndim"] == 3, v["arg1_ndim"] == 4)), v["arg2_value"] > 0), v["arg3_value"] > 0), v["arg4_value"] >= 0), v["arg5_length"] == 2), And([Implies(i < (1 + 1), And(And(Select(v["arg5_values"], i) > 0, v["arg4_value"] <= ((v["arg2_value"] - 1) * Select(v["arg5_values"], 0)) / 2), v["arg4_value"] <= ((v["arg2_value"] - 1) * Select(v["arg5_values"], 1)) / 2)) for i in range(6)]))) if n else
          And(And(And(And(And((Or(v["arg1_ndim"] == 3, v["arg1_ndim"] == 4)), v["arg2_value"] > 0), v["arg3_value"] > 0), v["arg4_value"] >= 0), v["arg5_length"] == 2), And([Implies(i < (1 + 1), And(And(Select(v["arg5_values"], i) > 0, v["arg4_value"] <= ((v["arg2_value"] - 1) * Select(v["arg5_values"], 0)) / 2), v["arg4_value"] <= ((v["arg2_value"] - 1) * Select(v["arg5_values"], 1)) / 2)) for i in range(6)])))
)

def rule_59_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False
        if not (isinstance(arg5, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg5)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')
        arg5_length = Int('arg5_length')
        arg5_values = Array('arg5_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == int(arg4))
        solver.add(arg5_length == len(arg5))
        for i in range(len(arg5)):
            arg5_values = Store(arg5_values, i, arg5[i])

        # Constraints for rule 59
        rule_59(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_length': arg5_length, 'arg5_values': arg5_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_59(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_length': arg5['length'], 'arg5_values': arg5['values']}, neg)
