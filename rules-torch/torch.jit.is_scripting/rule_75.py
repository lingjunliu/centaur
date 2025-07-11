import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if a float v_1 is greater than 0 and less than 1 and a boolean v_2 is true, then a tuple v_3 of int will have a length of 1 (Rule 75)

rule_75 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_value"] > 0, v["arg1_value"] < 1), v["arg2_value"] == True), v["arg3_length"] == 1, False)) if n else
          If(And(And(v["arg1_value"] > 0, v["arg1_value"] < 1), v["arg2_value"] == True), v["arg3_length"] == 1, False))
)

def rule_75_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, bool):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_length = Int('arg3_length')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_length == len(arg3))

        # Constraints for rule 75
        rule_75(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_75(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_length': arg3['length']}, neg)
