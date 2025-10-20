import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Input tensor cannot be of type bool, int8, int16, int32, int64, uint8 (Rule 45)

rule_45 = lambda s, v, n=False: (
    s.add(Not(Or([And(i < (6 + 1), v["arg1_dtype"] != i - 1) for i in range(6)])) if n else
          Or([And(i < (6 + 1), v["arg1_dtype"] != i - 1) for i in range(6)]))
)

def rule_45_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 45
        rule_45(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_45(solver, {'arg1_dtype': arg1['dtype']}, neg)
