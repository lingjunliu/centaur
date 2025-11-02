import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The fill_value type should be compatible with the tensor's dtype. (Rule 16)

rule_16 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 0, (Or(v["arg2_value"] == True, v["arg2_value"] == False)), If(And((v["arg1_dtype"] > 0), (v["arg1_dtype"] < 6)), True, If(And((v["arg1_dtype"] > 6), (v["arg1_dtype"] < 9)), True, True)))) if n else
          If(v["arg1_dtype"] == 0, (Or(v["arg2_value"] == True, v["arg2_value"] == False)), If(And((v["arg1_dtype"] > 0), (v["arg1_dtype"] < 6)), True, If(And((v["arg1_dtype"] > 6), (v["arg1_dtype"] < 9)), True, True))))
)

def rule_16_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, (float, np.floating)) or isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 16
        rule_16(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_16(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
