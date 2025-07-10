import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# logits dtype must be float16, float32, or float64, complex64, or complex128. checking !=0 to !=5 (Rule 93)

rule_93 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] != 0, If(v["arg1_dtype"] != 1, If(v["arg1_dtype"] != 2, If(v["arg1_dtype"] != 3, If(v["arg1_dtype"] != 4, If(v["arg1_dtype"] != 5, True, False), False), False), False), False), False)) if n else
          If(v["arg1_dtype"] != 0, If(v["arg1_dtype"] != 1, If(v["arg1_dtype"] != 2, If(v["arg1_dtype"] != 3, If(v["arg1_dtype"] != 4, If(v["arg1_dtype"] != 5, True, False), False), False), False), False), False))
)

def rule_93_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 93
        rule_93(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_93(solver, {'arg1_dtype': arg1['dtype']}, neg)
