import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# If the input tensor's dtype is float16, the output dtype cannot be bool, int8, int16, int32, int64, uint8 (Rule 17)

rule_17 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 6, And(And(And(And(And(v["arg2_dtype"] != 0, v["arg2_dtype"] != 1), v["arg2_dtype"] != 2), v["arg2_dtype"] != 3), v["arg2_dtype"] != 4), v["arg2_dtype"] != 5), False)) if n else
          If(v["arg1_dtype"] == 6, And(And(And(And(And(v["arg2_dtype"] != 0, v["arg2_dtype"] != 1), v["arg2_dtype"] != 2), v["arg2_dtype"] != 3), v["arg2_dtype"] != 4), v["arg2_dtype"] != 5), False))
)

def rule_17_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 17
        rule_17(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_17(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype']}, neg)
