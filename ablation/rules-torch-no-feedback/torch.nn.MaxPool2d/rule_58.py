import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# kernel_size as tuple(int (Rule 58)

rule_58 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_length"] == 2, And([Implies(i < (1 + 1), And(And(And(And(And(And(Select(v["arg1_values"], i) > 0, v["arg2_value"] > 0), v["arg3_value"] >= 0), v["arg4_value"] > 0), (Or(v["arg5_ndim"] == 3, v["arg5_ndim"] == 4))), v["arg3_value"] <= ((Select(v["arg1_values"], 0) - 1) * v["arg4_value"]) / 2), v["arg3_value"] <= ((Select(v["arg1_values"], 1) - 1) * v["arg4_value"]) / 2)) for i in range(6)]))) if n else
          And(v["arg1_length"] == 2, And([Implies(i < (1 + 1), And(And(And(And(And(And(Select(v["arg1_values"], i) > 0, v["arg2_value"] > 0), v["arg3_value"] >= 0), v["arg4_value"] > 0), (Or(v["arg5_ndim"] == 3, v["arg5_ndim"] == 4))), v["arg3_value"] <= ((Select(v["arg1_values"], 0) - 1) * v["arg4_value"]) / 2), v["arg3_value"] <= ((Select(v["arg1_values"], 1) - 1) * v["arg4_value"]) / 2)) for i in range(6)])))
)

def rule_58_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False
        if not isinstance(arg5, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')
        arg5_ndim = Int('arg5_ndim')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == int(arg4))
        solver.add(arg5_ndim == arg5.ndim)

        # Constraints for rule 58
        rule_58(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_ndim': arg5_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_58(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_ndim': arg5['ndim']}, neg)
