import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# checks alpha is within a reasonable range, depending on the datatype (Rule 115)

rule_115 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_dtype"] == 6, v["arg1_value"] < 100, If(v["arg2_dtype"] == 7, v["arg1_value"] < 10000, If(v["arg2_dtype"] == 8, v["arg1_value"] < 1000000, True)))) if n else
          If(v["arg2_dtype"] == 6, v["arg1_value"] < 100, If(v["arg2_dtype"] == 7, v["arg1_value"] < 10000, If(v["arg2_dtype"] == 8, v["arg1_value"] < 1000000, True))))
)

def rule_115_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 115
        rule_115(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_115(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype']}, neg)
