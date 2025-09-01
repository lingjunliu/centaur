import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Combined rule: Padding length constraints, valid mode, non-negative padding, mode replicate value, non-constant dimension check and invalid pad mode check (Rule 51)

rule_51 = lambda s, v, n=False: (
    s.add(Not(And(And((And(v["arg2_length"] <= 2 * v["arg1_ndim"], v["arg2_length"] % 2 == 0)), (Or(Or(Or(v["arg3_value"] == 21, v["arg3_value"] == 22), v["arg3_value"] == 23), v["arg3_value"] == 24))), And([Implies(i < (v["arg2_length"] - 1 + 1), And(And(And(Select(v["arg2_values"], i) >= 0, (If(v["arg3_value"] == 23, v["arg4_value"] == 0, True))), (If(v["arg3_value"] == 21, True, And(v["arg1_ndim"] >= 2, v["arg1_ndim"] <= 5)))), (Or(Or(Or(v["arg3_value"] == 21, v["arg3_value"] == 22), v["arg3_value"] == 23), v["arg3_value"] == 24)))) for i in range(6)]))) if n else
          And(And((And(v["arg2_length"] <= 2 * v["arg1_ndim"], v["arg2_length"] % 2 == 0)), (Or(Or(Or(v["arg3_value"] == 21, v["arg3_value"] == 22), v["arg3_value"] == 23), v["arg3_value"] == 24))), And([Implies(i < (v["arg2_length"] - 1 + 1), And(And(And(Select(v["arg2_values"], i) >= 0, (If(v["arg3_value"] == 23, v["arg4_value"] == 0, True))), (If(v["arg3_value"] == 21, True, And(v["arg1_ndim"] >= 2, v["arg1_ndim"] <= 5)))), (Or(Or(Or(v["arg3_value"] == 21, v["arg3_value"] == 22), v["arg3_value"] == 23), v["arg3_value"] == 24)))) for i in range(6)])))
)

def rule_51_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_value = String('arg3_value')
        arg4_value = Real('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_value == list_of_string_values_torch.index(arg3))
        solver.add(arg4_value == arg4)

        # Constraints for rule 51
        rule_51(solver, {'arg1_ndim': arg1_ndim, 'arg2_length': arg2_length, 'arg2_values': arg2_values, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_51(solver, {'arg1_ndim': arg1['ndim'], 'arg2_length': arg2['length'], 'arg2_values': arg2['values'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
