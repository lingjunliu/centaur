import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# input tensor must have a numeric/complex dtype: indices 0–10 (Rule 5)

rule_5 = lambda s, v, n=False: (
    s.add(Not(And(0 <= v["arg1_dtype"], v["arg1_dtype"] <= 10)) if n else
          And(0 <= v["arg1_dtype"], v["arg1_dtype"] <= 10))
)

def rule_5_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 5
        rule_5(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_5(solver, {'arg1_dtype': arg1['dtype']}, neg)
