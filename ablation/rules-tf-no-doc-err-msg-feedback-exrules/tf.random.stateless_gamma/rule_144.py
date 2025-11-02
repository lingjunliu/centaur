import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the dtype is a complex data type, alpha must also be a complex data type. (Rule 144)

rule_144 = lambda s, v, n=False: (
    s.add(Not(If((Or(v["arg2_value"] == 9, v["arg2_value"] == 10)), (Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10)), True)) if n else
          If((Or(v["arg2_value"] == 9, v["arg2_value"] == 10)), (Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10)), True))
)

def rule_144_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 144
        rule_144(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_144(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
