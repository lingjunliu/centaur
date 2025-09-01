import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the input tensor's values are integers, ensure that the result after the reciprocal operation will not lead to precision loss. e.g., use int32 for results requiring less precision (Rule 85)

rule_85 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 1, v["arg1_dtype"] == 7, If(v["arg1_dtype"] == 2, v["arg1_dtype"] == 7, If(v["arg1_dtype"] == 3, v["arg1_dtype"] == 8, True)))) if n else
          If(v["arg1_dtype"] == 1, v["arg1_dtype"] == 7, If(v["arg1_dtype"] == 2, v["arg1_dtype"] == 7, If(v["arg1_dtype"] == 3, v["arg1_dtype"] == 8, True))))
)

def rule_85_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 85
        rule_85(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_85(solver, {'arg1_dtype': arg1['dtype']}, neg)
