import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# dtype of indices and segment_ids should be int32 or int64 and data must be float or int and less than 11 (Rule 75)

rule_75 = lambda s, v, n=False: (
    s.add(Not(And(And(And((Or(v["arg1_dtype"] == 3, v["arg1_dtype"] == 4)), (Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4))), (Or(Or(Or(Or(Or(Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8), v["arg3_dtype"] == 1), v["arg3_dtype"] == 2), v["arg3_dtype"] == 3), v["arg3_dtype"] == 4), v["arg3_dtype"] == 5))), v["arg3_dtype"] < 11)) if n else
          And(And(And((Or(v["arg1_dtype"] == 3, v["arg1_dtype"] == 4)), (Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4))), (Or(Or(Or(Or(Or(Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8), v["arg3_dtype"] == 1), v["arg3_dtype"] == 2), v["arg3_dtype"] == 3), v["arg3_dtype"] == 4), v["arg3_dtype"] == 5))), v["arg3_dtype"] < 11))
)

def rule_75_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 75
        rule_75(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_75(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg3_dtype': arg3['dtype']}, neg)
