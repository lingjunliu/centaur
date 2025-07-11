import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the values tensor's dtype is int32 or int64, and binary_output is true, then the output's dtype is bool. (Rule 70)

rule_70 = lambda s, v, n=False: (
    s.add(Not(If(And((Or(v["arg1_dtype"] == 2, v["arg1_dtype"] == 3)), v["arg2_value"] == True), v["arg3_dtype"] == 0, False)) if n else
          If(And((Or(v["arg1_dtype"] == 2, v["arg1_dtype"] == 3)), v["arg2_value"] == True), v["arg3_dtype"] == 0, False))
)

def rule_70_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Bool('arg2_value')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 70
        rule_70(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_70(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype']}, neg)
