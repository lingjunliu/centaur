import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if string v_1 is 'tanh' then float v_2 must be in range -1 to 1 (Rule 61)

rule_61 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 11, And(v["arg2_value"] >= -1, v["arg2_value"] <= 1), False)) if n else
          If(v["arg1_value"] == 11, And(v["arg2_value"] >= -1, v["arg2_value"] <= 1), False))
)

def rule_61_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_value = Real('arg2_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values.index(arg1))
        solver.add(arg2_value == arg2)

        # Constraints for rule 61
        rule_61(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_61(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
