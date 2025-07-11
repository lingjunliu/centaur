import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If exclusive is true, and reverse is true, then the last element in output tensor will be zero. (Rule 25)

rule_25 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_value"] == True, v["arg3_value"] == True), True, False)) if n else
          If(And(v["arg2_value"] == True, v["arg3_value"] == True), True, False))
)

def rule_25_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg2_value = Bool('arg2_value')
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)

        # Constraints for rule 25
        rule_25(solver, {'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_25(solver, {'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
