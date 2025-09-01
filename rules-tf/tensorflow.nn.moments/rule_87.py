import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If shift is defined, then x must be defined and of type float, double, int32, uint8, int16, int8, complex64, int64, qint8, quint8, qint32, bfloat16, qint16, quint16, uint16, complex128, half, uint32, uint64. (Rule 87)

rule_87 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_ndim"] >= 0, (Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 3), v["arg1_dtype"] == 5), v["arg1_dtype"] == 2), v["arg1_dtype"] == 1), v["arg1_dtype"] == 10)), True)) if n else
          If(v["arg2_ndim"] >= 0, (Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 3), v["arg1_dtype"] == 5), v["arg1_dtype"] == 2), v["arg1_dtype"] == 1), v["arg1_dtype"] == 10)), True))
)

def rule_87_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 87
        rule_87(solver, {'arg1_dtype': arg1_dtype, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_87(solver, {'arg1_dtype': arg1['dtype'], 'arg2_ndim': arg2['ndim']}, neg)
