import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If axis is a tuple, then ord can not be int > 2  (Rule 95)

rule_95 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_length"] == 2, And(And(And(And(And(And(And((v["arg1_value"] != 3), (v["arg1_value"] != 4)), (v["arg1_value"] != 5)), (v["arg1_value"] != 6)), (v["arg1_value"] != 7)), (v["arg1_value"] != 8)), (v["arg1_value"] != 9)), (v["arg1_value"] != 10)), False)) if n else
          If(v["arg2_length"] == 2, And(And(And(And(And(And(And((v["arg1_value"] != 3), (v["arg1_value"] != 4)), (v["arg1_value"] != 5)), (v["arg1_value"] != 6)), (v["arg1_value"] != 7)), (v["arg1_value"] != 8)), (v["arg1_value"] != 9)), (v["arg1_value"] != 10)), False))
)

def rule_95_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, str)):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg2_length = Int('arg2_length')

        # Value assignments
        solver.add(arg2_length == len(arg2))

        # Constraints for rule 95
        rule_95(solver, {'arg1_value': arg1_value, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_95(solver, {'arg1_value': arg1['value'], 'arg2_length': arg2['length']}, neg)
