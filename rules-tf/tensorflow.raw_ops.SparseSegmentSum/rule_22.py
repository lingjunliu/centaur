import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if data is bfloat16 or half, then sparse gradient must be false (Rule 22)

rule_22 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg1_dtype"] == 14, v["arg1_dtype"] == 15), v["arg2_value"] == False, True)) if n else
          If(Or(v["arg1_dtype"] == 14, v["arg1_dtype"] == 15), v["arg2_value"] == False, True))
)

def rule_22_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Bool('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)

        # Constraints for rule 22
        rule_22(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_22(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
