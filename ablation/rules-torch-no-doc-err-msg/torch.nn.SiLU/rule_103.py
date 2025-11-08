import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# beta must be in a reasonable range to avoid underflow or overflow, also check for scale version 2  (Rule 103)

rule_103 = lambda s, v, n=False: (
    s.add(Not(If(And(-100 < v["arg1_value"], v["arg1_value"] < 100), And(-100 < v["arg2_value"], v["arg2_value"] < 100), True)) if n else
          If(And(-100 < v["arg1_value"], v["arg1_value"] < 100), And(-100 < v["arg2_value"], v["arg2_value"] < 100), True))
)

def rule_103_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 103
        rule_103(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_103(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
