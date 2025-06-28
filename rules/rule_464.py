import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if string v_1 is equal to tanh and float number v_2 is greater or equal to 0, then max of tensor v_3 should be smaller than 1000 and larger than float v_2 (Rule 464)

rule_464 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == 11, v["arg2_value"] >= 0), And(Select(v["arg3_range"], 1) < 1000, Select(v["arg3_range"], 1) > v["arg2_value"]), False)) if n else
          If(And(v["arg1_value"] == 11, v["arg2_value"] >= 0), And(Select(v["arg3_range"], 1) < 1000, Select(v["arg3_range"], 1) > v["arg2_value"]), False))
)

def rule_464_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_value = Real('arg2_value')
        arg3_range = Array('arg3_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values.index(arg1))
        solver.add(arg2_value == arg2)
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 464
        rule_464(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_range': arg3_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_464(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_range': arg3['range']}, neg)
