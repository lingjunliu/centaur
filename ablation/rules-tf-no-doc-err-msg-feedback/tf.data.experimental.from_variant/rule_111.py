import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Float v_1 must be greater than both minimum and maximum values of tensor v_2 if tuple v_3 contains only positive integers (Rule 111)

rule_111 = lambda s, v, n=False: (
    s.add(Not(If(And([Implies(i < (v["arg3_length"] - 1 + 1), Select(v["arg3_values"], i) > 0) for i in range(6)]), And(v["arg1_value"] > Select(v["arg2_range"], 1), v["arg1_value"] > Select(v["arg2_range"], 0)), True)) if n else
          If(And([Implies(i < (v["arg3_length"] - 1 + 1), Select(v["arg3_values"], i) > 0) for i in range(6)]), And(v["arg1_value"] > Select(v["arg2_range"], 1), v["arg1_value"] > Select(v["arg2_range"], 0)), True))
)

def rule_111_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 111
        rule_111(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range, 'arg3_values': arg3_values, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_111(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range'], 'arg3_values': arg3['values'], 'arg3_length': arg3['length']}, neg)
