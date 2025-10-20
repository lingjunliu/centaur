import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If one of the input is bfloat16, half, float32, float64, int8, uint8, int16, uint16, int32, uint32, int64, uint64, the other also has to be one of them. (Rule 41)

rule_41 = lambda s, v, n=False: (
    s.add(Not(And((Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 0, v["arg1_dtype"] == 6), v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 1), v["arg1_dtype"] == 5), v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4)), (Or(Or(Or(Or(Or(Or(Or(Or(v["arg2_dtype"] == 0, v["arg2_dtype"] == 6), v["arg2_dtype"] == 7), v["arg2_dtype"] == 8), v["arg2_dtype"] == 1), v["arg2_dtype"] == 5), v["arg2_dtype"] == 2), v["arg2_dtype"] == 3), v["arg2_dtype"] == 4)))) if n else
          And((Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 0, v["arg1_dtype"] == 6), v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 1), v["arg1_dtype"] == 5), v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4)), (Or(Or(Or(Or(Or(Or(Or(Or(v["arg2_dtype"] == 0, v["arg2_dtype"] == 6), v["arg2_dtype"] == 7), v["arg2_dtype"] == 8), v["arg2_dtype"] == 1), v["arg2_dtype"] == 5), v["arg2_dtype"] == 2), v["arg2_dtype"] == 3), v["arg2_dtype"] == 4))))
)

def rule_41_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 41
        rule_41(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_41(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype']}, neg)
