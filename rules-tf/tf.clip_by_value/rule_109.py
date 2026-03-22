import numpy as np
import torch
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# parameter 'x's DataType should not be bool; must be in list of allowed values:
# bfloat16, float16, float32, float64, int8, uint8, int16, uint16, int32, uint32, int64, uint64
# From list_of_available_dtypes, allowed indices: 1 (int8), 2 (int16), 3 (int32), 4 (int64),
# 5 (uint8), 6 (float16), 7 (float32), 8 (float64), 9 (complex64), 10 (complex128)

rule_109 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 5), v["arg1_dtype"] == 6), v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10)) if n else
          Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 5), v["arg1_dtype"] == 6), v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10))
)

def rule_109_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        solver = Solver()
        arg1_dtype = Int('arg1_dtype')

        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        rule_109(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_109(solver, {'arg1_dtype': arg1['dtype']}, neg)