import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The generated tensor dtype is same as specified dtype (Rule 35)

rule_35 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 7, v["arg2_dtype"] == 7, If(v["arg1_value"] == 8, v["arg2_dtype"] == 8, If(v["arg1_value"] == 9, v["arg2_dtype"] == 9, False)))) if n else
          If(v["arg1_value"] == 7, v["arg2_dtype"] == 7, If(v["arg1_value"] == 8, v["arg2_dtype"] == 8, If(v["arg1_value"] == 9, v["arg2_dtype"] == 9, False))))
)

def rule_35_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 35
        rule_35(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_35(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype']}, neg)
