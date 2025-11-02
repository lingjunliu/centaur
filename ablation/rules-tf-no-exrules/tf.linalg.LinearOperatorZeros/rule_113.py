import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If matrix is not square restrict num_rows to a range if num_columns is specified and smaller than num_rows (Rule 113)

rule_113 = lambda s, v, n=False: (
    s.add(Not(If(And(And((False == v["arg3_value"]), v["arg2_value"] > 0), v["arg2_value"] < v["arg1_value"]), And(v["arg1_value"] > 10, v["arg1_value"] < 2000), True)) if n else
          If(And(And((False == v["arg3_value"]), v["arg2_value"] > 0), v["arg2_value"] < v["arg1_value"]), And(v["arg1_value"] > 10, v["arg1_value"] < 2000), True))
)

def rule_113_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == arg3)

        # Constraints for rule 113
        rule_113(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_113(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
