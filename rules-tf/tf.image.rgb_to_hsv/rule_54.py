import numpy as np
import torch
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# parameter 'images' DataType must be in: float16, bfloat16, float32, float64
# From list_of_available_dtypes, allowed indices: 6 (float16), 7 (float32), 8 (float64)
# Excluded: 0 (bool), 1 (int8), 2 (int16), 3 (int32), 4 (int64), 5 (uint8),
#           9 (complex64), 10 (complex128)

rule_54 = lambda s, v, n=False: (
    s.add(Not(Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8))) if n else
    s.add(Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8))
)

def rule_54_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        solver = Solver()
        arg1_dtype = Int('arg1_dtype')

        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        rule_54(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_54(solver, {'arg1_dtype': arg1['dtype']}, neg)