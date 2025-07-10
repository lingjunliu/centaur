import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If the input tensor's dtype is complex, the output tensor's dtype should also be complex if axis is None (Rule 87)

rule_87 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg1_dtype"] == 10, v["arg1_dtype"] == 11), True, False)) if n else
          If(Or(v["arg1_dtype"] == 10, v["arg1_dtype"] == 11), True, False))
)

def rule_87_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 87
        rule_87(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_87(solver, {'arg1_dtype': arg1['dtype']}, neg)
