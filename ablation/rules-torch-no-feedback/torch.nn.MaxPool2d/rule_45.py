import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Input is Tensor, kernel_size as tuple(int (Rule 45)

rule_45 = lambda s, v, n=False: (
    s.add(Not(And(And((Or(v["arg1_ndim"] == 3, v["arg1_ndim"] == 4)), v["arg2_length"] == 2), And([Implies(i < (1 + 1), And(And(And(Select(v["arg2_values"], i) > 0, v["arg3_value"] > 0), v["arg4_length"] == 2), And([Implies(i < (1 + 1), And(And(Select(v["arg4_values"], i) >= 0, v["arg5_value"] > 0), And([Implies(i < (1 + 1), Select(v["arg4_values"], i) <= ((Select(v["arg2_values"], i) - 1) * v["arg5_value"]) / 2) for i in range(6)]))) for i in range(6)]))) for i in range(6)]))) if n else
          And(And((Or(v["arg1_ndim"] == 3, v["arg1_ndim"] == 4)), v["arg2_length"] == 2), And([Implies(i < (1 + 1), And(And(And(Select(v["arg2_values"], i) > 0, v["arg3_value"] > 0), v["arg4_length"] == 2), And([Implies(i < (1 + 1), And(And(Select(v["arg4_values"], i) >= 0, v["arg5_value"] > 0), And([Implies(i < (1 + 1), Select(v["arg4_values"], i) <= ((Select(v["arg2_values"], i) - 1) * v["arg5_value"]) / 2) for i in range(6)]))) for i in range(6)]))) for i in range(6)])))
)

def rule_45_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg4, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False
        if not (isinstance(arg5, (int, np.integer)) and not isinstance(arg5, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_value = Int('arg3_value')
        arg4_length = Int('arg4_length')
        arg4_values = Array('arg4_values', IntSort(), IntSort())
        arg5_value = Int('arg5_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_length == len(arg4))
        for i in range(len(arg4)):
            arg4_values = Store(arg4_values, i, arg4[i])
        solver.add(arg5_value == int(arg5))

        # Constraints for rule 45
        rule_45(solver, {'arg1_ndim': arg1_ndim, 'arg2_length': arg2_length, 'arg2_values': arg2_values, 'arg3_value': arg3_value, 'arg4_length': arg4_length, 'arg4_values': arg4_values, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_45(solver, {'arg1_ndim': arg1['ndim'], 'arg2_length': arg2['length'], 'arg2_values': arg2['values'], 'arg3_value': arg3['value'], 'arg4_length': arg4['length'], 'arg4_values': arg4['values'], 'arg5_value': arg5['value']}, neg)
