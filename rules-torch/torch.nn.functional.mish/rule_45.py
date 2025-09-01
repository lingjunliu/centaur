import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The input tensor cannot be bool type, if inplace is True, then the minimum value of the tensor must be small than maximum value (Rule 45)

rule_45 = lambda s, v, n=False: (
    s.add(Not(And((v["arg1_dtype"] != 0), (If(v["arg2_value"] == True, Select(v["arg1_range"], 0) < Select(v["arg1_range"], 1), True)))) if n else
          And((v["arg1_dtype"] != 0), (If(v["arg2_value"] == True, Select(v["arg1_range"], 0) < Select(v["arg1_range"], 1), True))))
)

def rule_45_func(arg1, arg2, solver=None, neg=False):
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
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Bool('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == arg2)

        # Constraints for rule 45
        rule_45(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_45(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
