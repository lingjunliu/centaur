import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If input dtype is float32, then output dtype should be float32, else if input is bfloat16, output is bfloat16, else if input is float16, output is float16, else output is float64 (Rule 29)

rule_29 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 7, v["arg1_dtype"] == 7, If(v["arg1_dtype"] == 13, v["arg1_dtype"] == 13, If(v["arg1_dtype"] == 6, v["arg1_dtype"] == 6, v["arg1_dtype"] == 8)))) if n else
          If(v["arg1_dtype"] == 7, v["arg1_dtype"] == 7, If(v["arg1_dtype"] == 13, v["arg1_dtype"] == 13, If(v["arg1_dtype"] == 6, v["arg1_dtype"] == 6, v["arg1_dtype"] == 8))))
)

def rule_29_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 29
        rule_29(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_29(solver, {'arg1_dtype': arg1['dtype']}, neg)
