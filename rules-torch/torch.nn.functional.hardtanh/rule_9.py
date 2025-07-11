import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if max_val is positive and min_val is positive, min_val has to be less than or equal to max_val (Rule 9)

rule_9 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] > 0, v["arg2_value"] > 0), v["arg1_value"] <= v["arg2_value"], False)) if n else
          If(And(v["arg1_value"] > 0, v["arg2_value"] > 0), v["arg1_value"] <= v["arg2_value"], False))
)

def rule_9_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Real('arg2_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)

        # Constraints for rule 9
        rule_9(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_9(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
