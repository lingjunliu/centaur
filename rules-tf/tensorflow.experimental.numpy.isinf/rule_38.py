import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# isinf only supports float16, float32, float64, complex64, and complex128 (Rule 38)

rule_38 = lambda s, v, n=False: (
    s.add(Not(If((v["arg1_dtype"] < 6), False, If((v["arg1_dtype"] > 10), False, True))) if n else
          If((v["arg1_dtype"] < 6), False, If((v["arg1_dtype"] > 10), False, True)))
)

def rule_38_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 38
        rule_38(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_38(solver, {'arg1_dtype': arg1['dtype']}, neg)
