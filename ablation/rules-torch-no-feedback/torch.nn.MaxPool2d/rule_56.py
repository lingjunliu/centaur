import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# kernel_size, stride as tuple(int (Rule 56)

rule_56 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(v["arg1_length"] == 2, v["arg2_length"] == 2), v["arg3_length"] == 2), v["arg4_length"] == 2), And([Implies(i < (1 + 1), And(Select(v["arg1_values"], i) > 0, And([Implies(i < (1 + 1), And(Select(v["arg2_values"], i) > 0, And([Implies(i < (1 + 1), And(Select(v["arg3_values"], i) >= 0, And([Implies(i < (1 + 1), And(And(Select(v["arg4_values"], i) > 0, (Or(v["arg5_ndim"] == 3, v["arg5_ndim"] == 4))), And([Implies(i < (1 + 1), Select(v["arg3_values"], i) <= (Select(v["arg1_values"], i) - 1) * Select(v["arg4_values"], i) / 2) for i in range(6)]))) for i in range(6)]))) for i in range(6)]))) for i in range(6)]))) for i in range(6)]))) if n else
          And(And(And(And(v["arg1_length"] == 2, v["arg2_length"] == 2), v["arg3_length"] == 2), v["arg4_length"] == 2), And([Implies(i < (1 + 1), And(Select(v["arg1_values"], i) > 0, And([Implies(i < (1 + 1), And(Select(v["arg2_values"], i) > 0, And([Implies(i < (1 + 1), And(Select(v["arg3_values"], i) >= 0, And([Implies(i < (1 + 1), And(And(Select(v["arg4_values"], i) > 0, (Or(v["arg5_ndim"] == 3, v["arg5_ndim"] == 4))), And([Implies(i < (1 + 1), Select(v["arg3_values"], i) <= (Select(v["arg1_values"], i) - 1) * Select(v["arg4_values"], i) / 2) for i in range(6)]))) for i in range(6)]))) for i in range(6)]))) for i in range(6)]))) for i in range(6)])))
)

def rule_56_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False
        if not (isinstance(arg4, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False
        if not isinstance(arg5, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())
        arg4_length = Int('arg4_length')
        arg4_values = Array('arg4_values', IntSort(), IntSort())
        arg5_ndim = Int('arg5_ndim')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])
        solver.add(arg4_length == len(arg4))
        for i in range(len(arg4)):
            arg4_values = Store(arg4_values, i, arg4[i])
        solver.add(arg5_ndim == arg5.ndim)

        # Constraints for rule 56
        rule_56(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values, 'arg2_length': arg2_length, 'arg2_values': arg2_values, 'arg3_length': arg3_length, 'arg3_values': arg3_values, 'arg4_length': arg4_length, 'arg4_values': arg4_values, 'arg5_ndim': arg5_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_56(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values'], 'arg2_length': arg2['length'], 'arg2_values': arg2['values'], 'arg3_length': arg3['length'], 'arg3_values': arg3['values'], 'arg4_length': arg4['length'], 'arg4_values': arg4['values'], 'arg5_ndim': arg5['ndim']}, neg)
