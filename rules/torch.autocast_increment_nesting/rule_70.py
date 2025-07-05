import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If autocast is False and dtype is not integer, then increment can be 0 or equal to -1 or -2 (Rule 70)

rule_70 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == False, (Or(v["arg3_value"] < 1, v["arg3_value"] > 5))), Or(Or(v["arg2_value"] == 0, v["arg2_value"] == -1), v["arg2_value"] == -2), False)) if n else
          If(And(v["arg1_value"] == False, (Or(v["arg3_value"] < 1, v["arg3_value"] > 5))), Or(Or(v["arg2_value"] == 0, v["arg2_value"] == -1), v["arg2_value"] == -2), False))
)

def rule_70_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, torch.dtype) or isinstance(arg3, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == list_of_available_dtypes.index(np_dtype(arg3)))

        # Constraints for rule 70
        rule_70(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_70(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
