import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# String must be in a given set if dtype is 7. (Rule 131)

rule_131 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == 7, Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_value"] == 11, v["arg1_value"] == 12), v["arg1_value"] == 13), v["arg1_value"] == 14), v["arg1_value"] == 15), v["arg1_value"] == 16), v["arg1_value"] == 17), v["arg1_value"] == 18), v["arg1_value"] == 19), True)) if n else
          If(v["arg2_value"] == 7, Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_value"] == 11, v["arg1_value"] == 12), v["arg1_value"] == 13), v["arg1_value"] == 14), v["arg1_value"] == 15), v["arg1_value"] == 16), v["arg1_value"] == 17), v["arg1_value"] == 18), v["arg1_value"] == 19), True))
)

def rule_131_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))

        # Constraints for rule 131
        rule_131(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_131(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
