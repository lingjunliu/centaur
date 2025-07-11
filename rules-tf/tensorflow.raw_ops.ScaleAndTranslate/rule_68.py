import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If images is complex64/complex128/bfloat16/half, antialias must be false (Rule 68)

rule_68 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(Or(v["arg1_dtype"] == 10, v["arg1_dtype"] == 11), v["arg1_dtype"] == 7), v["arg1_dtype"] == 6), v["arg2_value"] == False, False)) if n else
          If(Or(Or(Or(v["arg1_dtype"] == 10, v["arg1_dtype"] == 11), v["arg1_dtype"] == 7), v["arg1_dtype"] == 6), v["arg2_value"] == False, False))
)

def rule_68_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 68
        rule_68(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_68(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
