import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# var must be a float32, float64, int32, uint8, int16, int8, complex64, int64, qint8, quint8, qint32, bfloat16, qint16, quint16, uint16, complex128, half, uint32, uint64 tensor (Rule 9)

rule_9 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 3), v["arg1_dtype"] == 5), v["arg1_dtype"] == 2), v["arg1_dtype"] == 1), v["arg1_dtype"] == 9), v["arg1_dtype"] == 4), v["arg1_dtype"] == 13), v["arg1_dtype"] == 14), v["arg1_dtype"] == 15), v["arg1_dtype"] == 6), v["arg1_dtype"] == 16), v["arg1_dtype"] == 17)) if n else
          Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 3), v["arg1_dtype"] == 5), v["arg1_dtype"] == 2), v["arg1_dtype"] == 1), v["arg1_dtype"] == 9), v["arg1_dtype"] == 4), v["arg1_dtype"] == 13), v["arg1_dtype"] == 14), v["arg1_dtype"] == 15), v["arg1_dtype"] == 6), v["arg1_dtype"] == 16), v["arg1_dtype"] == 17))
)

def rule_9_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 9
        rule_9(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_9(solver, {'arg1_dtype': arg1['dtype']}, neg)
