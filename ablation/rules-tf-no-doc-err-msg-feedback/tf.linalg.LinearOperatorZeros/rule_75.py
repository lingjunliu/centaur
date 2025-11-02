import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If is_non_singular is specified to be True or False, then the given shape should be valid. (Rule 75)

rule_75 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg1_value"] == True, v["arg1_value"] == False), And(And(v["arg2_value"] > 0, v["arg3_value"] > 0), And([Implies(i < (v["arg4_length"] - 1 + 1), Select(v["arg4_values"], i) > 0) for i in range(6)])), True)) if n else
          If(Or(v["arg1_value"] == True, v["arg1_value"] == False), And(And(v["arg2_value"] > 0, v["arg3_value"] > 0), And([Implies(i < (v["arg4_length"] - 1 + 1), Select(v["arg4_values"], i) > 0) for i in range(6)])), True))
)

def rule_75_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg4, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')
        arg4_length = Int('arg4_length')
        arg4_values = Array('arg4_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_length == len(arg4))
        for i in range(len(arg4)):
            arg4_values = Store(arg4_values, i, arg4[i])

        # Constraints for rule 75
        rule_75(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_values': arg4_values, 'arg4_length': arg4_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_75(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_values': arg4['values'], 'arg4_length': arg4['length']}, neg)
