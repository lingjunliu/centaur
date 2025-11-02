import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# padding int vs effective kernel per-dimension: kernel_size tuple, dilation int (Rule 12)

rule_12 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg1_length"] == 3, v["arg2_value"] >= 1), v["arg3_value"] >= 0), And([Implies(i < (v["arg1_length"] - 1 + 1), 2 * v["arg3_value"] <= v["arg2_value"] * (Select(v["arg1_values"], i) - 1) + 1) for i in range(6)]))) if n else
          And(And(And(v["arg1_length"] == 3, v["arg2_value"] >= 1), v["arg3_value"] >= 0), And([Implies(i < (v["arg1_length"] - 1 + 1), 2 * v["arg3_value"] <= v["arg2_value"] * (Select(v["arg1_values"], i) - 1) + 1) for i in range(6)])))
)

def rule_12_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 12
        rule_12(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_12(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
