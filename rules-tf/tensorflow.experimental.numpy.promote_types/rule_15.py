import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Promoting an int with a complex gives a complex (Rule 15)

rule_15 = lambda s, v, n=False: (
    s.add(Not(If((And((And(1 <= v["arg1_value"], v["arg1_value"] <= 5)), (And(9 <= v["arg2_value"], v["arg2_value"] <= 10)))), True, If((And((And(1 <= v["arg2_value"], v["arg2_value"] <= 5)), (And(9 <= v["arg1_value"], v["arg1_value"] <= 10)))), True, False))) if n else
          If((And((And(1 <= v["arg1_value"], v["arg1_value"] <= 5)), (And(9 <= v["arg2_value"], v["arg2_value"] <= 10)))), True, If((And((And(1 <= v["arg2_value"], v["arg2_value"] <= 5)), (And(9 <= v["arg1_value"], v["arg1_value"] <= 10)))), True, False)))
)

def rule_15_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))

        # Constraints for rule 15
        rule_15(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_15(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
