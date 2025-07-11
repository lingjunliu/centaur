import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# lambd should not be a large value to prevent overflow when converting to Half type and tensor dtype should not be int8, int16, int32, int64, uint8 (Rule 6)

rule_6 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(v["arg1_value"] < 1000, v["arg2_dtype"] != 1), v["arg2_dtype"] != 2), v["arg2_dtype"] != 3), v["arg2_dtype"] != 4), v["arg2_dtype"] != 5)) if n else
          And(And(And(And(And(v["arg1_value"] < 1000, v["arg2_dtype"] != 1), v["arg2_dtype"] != 2), v["arg2_dtype"] != 3), v["arg2_dtype"] != 4), v["arg2_dtype"] != 5))
)

def rule_6_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 6
        rule_6(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_6(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype']}, neg)
