import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The indices tensor in `sparse_fill_empty_rows` must be type int64 or int32. (Rule 127)

rule_127 = lambda s, v, n=False: (
    s.add(Not(Or((v["arg1_dtype"] == 3), (v["arg1_dtype"] == 4))) if n else
          Or((v["arg1_dtype"] == 3), (v["arg1_dtype"] == 4)))
)

def rule_127_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 127
        rule_127(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_127(solver, {'arg1_dtype': arg1['dtype']}, neg)
