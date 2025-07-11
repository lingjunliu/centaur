import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Keys can't have dtype as bool, int8, int16, int32, int64, uint8, float16, float32, float64, complex64, complex128, dtype (Rule 36)

rule_36 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(And(And(And(v["arg1_dtype"] != 0, v["arg1_dtype"] != 1), v["arg1_dtype"] != 2), v["arg1_dtype"] != 3), v["arg1_dtype"] != 4), v["arg1_dtype"] != 5), v["arg1_dtype"] != 6), v["arg1_dtype"] != 7), v["arg1_dtype"] != 8), v["arg1_dtype"] != 9), v["arg1_dtype"] != 10), v["arg1_dtype"] != 13)) if n else
          And(And(And(And(And(And(And(And(And(And(And(v["arg1_dtype"] != 0, v["arg1_dtype"] != 1), v["arg1_dtype"] != 2), v["arg1_dtype"] != 3), v["arg1_dtype"] != 4), v["arg1_dtype"] != 5), v["arg1_dtype"] != 6), v["arg1_dtype"] != 7), v["arg1_dtype"] != 8), v["arg1_dtype"] != 9), v["arg1_dtype"] != 10), v["arg1_dtype"] != 13))
)

def rule_36_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 36
        rule_36(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_36(solver, {'arg1_dtype': arg1['dtype']}, neg)
