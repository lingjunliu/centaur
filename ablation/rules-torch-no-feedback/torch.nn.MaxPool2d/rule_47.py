import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Input is Tensor, kernel_size as int, stride as tuple(int (Rule 47)

rule_47 = lambda s, v, n=False: (
    s.add(Not(And(And(And((Or(v["arg1_ndim"] == 3, v["arg1_ndim"] == 4)), v["arg2_value"] > 0), v["arg3_length"] == 2), And([Implies(i < (1 + 1), And(And(Select(v["arg3_values"], i) > 0, v["arg4_length"] == 2), And([Implies(i < (1 + 1), And(And(Select(v["arg4_values"], i) >= 0, v["arg5_length"] == 2), And([Implies(i < (1 + 1), And(Select(v["arg5_values"], i) > 0, And([Implies(i < (1 + 1), Select(v["arg4_values"], i) <= ((v["arg2_value"] - 1) * Select(v["arg5_values"], i)) / 2) for i in range(6)]))) for i in range(6)]))) for i in range(6)]))) for i in range(6)]))) if n else
          And(And(And((Or(v["arg1_ndim"] == 3, v["arg1_ndim"] == 4)), v["arg2_value"] > 0), v["arg3_length"] == 2), And([Implies(i < (1 + 1), And(And(Select(v["arg3_values"], i) > 0, v["arg4_length"] == 2), And([Implies(i < (1 + 1), And(And(Select(v["arg4_values"], i) >= 0, v["arg5_length"] == 2), And([Implies(i < (1 + 1), And(Select(v["arg5_values"], i) > 0, And([Implies(i < (1 + 1), Select(v["arg4_values"], i) <= ((v["arg2_value"] - 1) * Select(v["arg5_values"], i)) / 2) for i in range(6)]))) for i in range(6)]))) for i in range(6)]))) for i in range(6)])))
)

def rule_47_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
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
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False
        if not (isinstance(arg4, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False
        if not (isinstance(arg5, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg5)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = Int('arg2_value')
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())
        arg4_length = Int('arg4_length')
        arg4_values = Array('arg4_values', IntSort(), IntSort())
        arg5_length = Int('arg5_length')
        arg5_values = Array('arg5_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])
        solver.add(arg4_length == len(arg4))
        for i in range(len(arg4)):
            arg4_values = Store(arg4_values, i, arg4[i])
        solver.add(arg5_length == len(arg5))
        for i in range(len(arg5)):
            arg5_values = Store(arg5_values, i, arg5[i])

        # Constraints for rule 47
        rule_47(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_length': arg3_length, 'arg3_values': arg3_values, 'arg4_length': arg4_length, 'arg4_values': arg4_values, 'arg5_length': arg5_length, 'arg5_values': arg5_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_47(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_length': arg3['length'], 'arg3_values': arg3['values'], 'arg4_length': arg4['length'], 'arg4_values': arg4['values'], 'arg5_length': arg5['length'], 'arg5_values': arg5['values']}, neg)
