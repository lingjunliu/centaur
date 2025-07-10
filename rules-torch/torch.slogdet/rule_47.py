import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Input A must have the correct dtype, it can not be bool, int8, int16, int32, int64, uint8 to prevent char error (Rule 47)

rule_47 = lambda s, v, n=False: (
    s.add(Not(And((v["arg1_dtype"] > 5), (v["arg1_dtype"] < 11))) if n else
          And((v["arg1_dtype"] > 5), (v["arg1_dtype"] < 11)))
)

def rule_47_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 47
        rule_47(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_47(solver, {'arg1_dtype': arg1['dtype']}, neg)
