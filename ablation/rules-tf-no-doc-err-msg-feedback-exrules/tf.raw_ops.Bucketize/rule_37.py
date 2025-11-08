import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if type_x is a union of int and float, then type_buckets must be float (Rule 37)

rule_37 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(Or(v["arg3_value"] == 3, v["arg3_value"] == 7), v["arg3_value"] == 4), v["arg3_value"] == 8), (If(Or(v["arg4_value"] == 7, v["arg4_value"] == 8), True, False)), True)) if n else
          If(Or(Or(Or(v["arg3_value"] == 3, v["arg3_value"] == 7), v["arg3_value"] == 4), v["arg3_value"] == 8), (If(Or(v["arg4_value"] == 7, v["arg4_value"] == 8), True, False)), True))
)

def rule_37_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == int(arg4))

        # Constraints for rule 37
        rule_37(solver, {'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_37(solver, {'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
