import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the output tensor is of type short, the input tensor should be non-complex, with minimum value greather than 0 and a maximum less than 60, and the data type should not be np.float32 or np.float64 (Rule 46)

rule_46 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_dtype"] == 2, And(And(And(And(And(v["arg1_dtype"] != 9, v["arg1_dtype"] != 10), Select(v["arg1_range"], 0) >= 0), Select(v["arg1_range"], 1) < 60), v["arg1_dtype"] != 7), v["arg1_dtype"] != 8), True)) if n else
          If(v["arg2_dtype"] == 2, And(And(And(And(And(v["arg1_dtype"] != 9, v["arg1_dtype"] != 10), Select(v["arg1_range"], 0) >= 0), Select(v["arg1_range"], 1) < 60), v["arg1_dtype"] != 7), v["arg1_dtype"] != 8), True))
)

def rule_46_func(arg1, arg2, solver=None, neg=False):
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
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 46
        rule_46(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_46(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range'], 'arg2_dtype': arg2['dtype']}, neg)
