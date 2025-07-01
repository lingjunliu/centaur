import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if the dtype is 0 and the string is 'sum','mean' or 'max', bool should be false (Rule 1426)

rule_1426 = lambda s, v, n=False: (
    s.add(Not(If(And((v["arg1_value"] == 0), (Or(Or(v["arg2_value"] == 8, v["arg2_value"] == 7), v["arg2_value"] == 9))), v["arg3_value"] == False, False)) if n else
          If(And((v["arg1_value"] == 0), (Or(Or(v["arg2_value"] == 8, v["arg2_value"] == 7), v["arg2_value"] == 9))), v["arg3_value"] == False, False))
)

def rule_1426_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not isinstance(arg2, str):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = String('arg2_value')
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_value == list_of_string_values.index(arg2))
        solver.add(arg3_value == arg3)

        # Constraints for rule 1426
        rule_1426(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_1426(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
