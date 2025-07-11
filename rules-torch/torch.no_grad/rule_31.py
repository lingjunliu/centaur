import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If a condition is true and a tensor's dtype is an integer, then a comparison holds. (Rule 31)

rule_31 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == True, (Or(Or(v["arg2_dtype"] == 1, v["arg2_dtype"] == 2), v["arg2_dtype"] == 3))), v["arg3_value"] > 0, False)) if n else
          If(And(v["arg1_value"] == True, (Or(Or(v["arg2_dtype"] == 1, v["arg2_dtype"] == 2), v["arg2_dtype"] == 3))), v["arg3_value"] > 0, False))
)

def rule_31_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_dtype = Int('arg2_dtype')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 31
        rule_31(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_31(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg3_value': arg3['value']}, neg)
