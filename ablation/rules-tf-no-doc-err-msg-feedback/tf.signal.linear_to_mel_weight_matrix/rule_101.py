import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Final check for relationships amongst key parameters (Rule 101)

rule_101 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(v["arg1_value"] > 0, v["arg2_value"] > 0), v["arg3_value"] > 0), v["arg4_value"] >= 0), v["arg5_value"] > v["arg4_value"]), v["arg5_value"] <= v["arg1_value"] / 2), v["arg2_value"] <= v["arg1_value"])) if n else
          And(And(And(And(And(And(v["arg1_value"] > 0, v["arg2_value"] > 0), v["arg3_value"] > 0), v["arg4_value"] >= 0), v["arg5_value"] > v["arg4_value"]), v["arg5_value"] <= v["arg1_value"] / 2), v["arg2_value"] <= v["arg1_value"]))
)

def rule_101_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, (float, np.floating))):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not ((isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)) or isinstance(arg4, (float, np.floating))):
            return False
        if not ((isinstance(arg5, (int, np.integer)) and not isinstance(arg5, bool)) or isinstance(arg5, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 101
        rule_101(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_101(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value']}, neg)
