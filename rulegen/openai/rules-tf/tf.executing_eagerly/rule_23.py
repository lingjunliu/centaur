import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# numeric depth ≤ 0 implies outside tf.function when not disabled and not in dataset => result is True (Rule 23)

rule_23 = lambda s, v, n=False: (
    s.add(Not(If(And(And((v["arg4_value"] <= 0), (v["arg2_value"] == False)), (v["arg3_value"] == False)), v["arg1_value"] == True, True)) if n else
          If(And(And((v["arg4_value"] <= 0), (v["arg2_value"] == False)), (v["arg3_value"] == False)), v["arg1_value"] == True, True))
)

def rule_23_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, bool):
            return False
        if not ((isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)) or isinstance(arg4, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)

        # Constraints for rule 23
        rule_23(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_23(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
