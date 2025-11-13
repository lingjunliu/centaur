import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# sample_rate should be less than or equal to the maximum allowable sample rate considering the data type (Rule 13)

rule_13 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_dtype"] == 6, v["arg1_value"] <= 96000, If(v["arg2_dtype"] == 7, v["arg1_value"] <= 192000, v["arg1_value"] <= 48000))) if n else
          If(v["arg2_dtype"] == 6, v["arg1_value"] <= 96000, If(v["arg2_dtype"] == 7, v["arg1_value"] <= 192000, v["arg1_value"] <= 48000)))
)

def rule_13_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 13
        rule_13(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_13(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype']}, neg)
