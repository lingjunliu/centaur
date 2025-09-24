import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# padding tuple vs effective kernel: kernel_size int, dilation int (Rule 11)

rule_11 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg1_value"] >= 1, v["arg2_value"] >= 1), v["arg3_length"] == 3), And([Implies(i < (v["arg3_length"] - 1 + 1), 2 * Select(v["arg3_values"], i) <= v["arg2_value"] * (v["arg1_value"] - 1) + 1) for i in range(6)]))) if n else
          And(And(And(v["arg1_value"] >= 1, v["arg2_value"] >= 1), v["arg3_length"] == 3), And([Implies(i < (v["arg3_length"] - 1 + 1), 2 * Select(v["arg3_values"], i) <= v["arg2_value"] * (v["arg1_value"] - 1) + 1) for i in range(6)])))
)

def rule_11_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 11
        rule_11(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_values': arg3_values, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_11(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_values': arg3['values'], 'arg3_length': arg3['length']}, neg)
