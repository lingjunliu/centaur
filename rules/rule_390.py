import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if string v_1 equals "constant", then integer v_2 must be in the range of int32 (-2147483648 to 2147483647 (Rule 390)

rule_390 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 10, And(v["arg2_value"] >= -2147483648, v["arg2_value"] <= 2147483647), False)) if n else
          If(v["arg1_value"] == 10, And(v["arg2_value"] >= -2147483648, v["arg2_value"] <= 2147483647), False))
)

def rule_390_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values.index(arg1))
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 390
        rule_390(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_390(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
