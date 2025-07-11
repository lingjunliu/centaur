import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If output_range_given is true, then given_y_min must be smaller or equal to given_y_max. (Rule 132)

rule_132 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == True, v["arg2_value"] <= v["arg3_value"], False)) if n else
          If(v["arg1_value"] == True, v["arg2_value"] <= v["arg3_value"], False))
)

def rule_132_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Real('arg2_value')
        arg3_value = Real('arg3_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)

        # Constraints for rule 132
        rule_132(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_132(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
