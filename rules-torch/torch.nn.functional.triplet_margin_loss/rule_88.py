import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If P > 0 and margin >0 then swap has to be a boolean value if the reduction is mean, sum or none (Rule 88)

rule_88 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] > 0, v["arg2_value"] > 0), (If((Or(Or(v["arg4_value"] == 8, v["arg4_value"] == 7), v["arg4_value"] == 6)), (Or(v["arg3_value"] == True, v["arg3_value"] == False)), False)), False)) if n else
          If(And(v["arg1_value"] > 0, v["arg2_value"] > 0), (If((Or(Or(v["arg4_value"] == 8, v["arg4_value"] == 7), v["arg4_value"] == 6)), (Or(v["arg3_value"] == True, v["arg3_value"] == False)), False)), False))
)

def rule_88_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, bool):
            return False
        if not isinstance(arg4, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Real('arg2_value')
        arg3_value = Bool('arg3_value')
        arg4_value = String('arg4_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == list_of_string_values.index(arg4))

        # Constraints for rule 88
        rule_88(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_88(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
