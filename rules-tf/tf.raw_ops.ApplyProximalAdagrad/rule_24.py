import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If var is uint8, int16, int8, qint8, quint8, qint32, bfloat16, qint16, quint16, uint16, uint32, uint64 the grad should have same type respectively (Rule 24)

rule_24 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 5, v["arg1_dtype"] == 2), v["arg1_dtype"] == 1), v["arg1_dtype"] == 0), v["arg1_dtype"] == 12), v["arg1_dtype"] == 13), v["arg1_dtype"] == 11), v["arg1_dtype"] == 14), v["arg1_dtype"] == 15), v["arg1_dtype"] == 16), v["arg1_dtype"] == 17), v["arg1_dtype"] == 4), v["arg2_dtype"] == v["arg1_dtype"], True)) if n else
          If(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 5, v["arg1_dtype"] == 2), v["arg1_dtype"] == 1), v["arg1_dtype"] == 0), v["arg1_dtype"] == 12), v["arg1_dtype"] == 13), v["arg1_dtype"] == 11), v["arg1_dtype"] == 14), v["arg1_dtype"] == 15), v["arg1_dtype"] == 16), v["arg1_dtype"] == 17), v["arg1_dtype"] == 4), v["arg2_dtype"] == v["arg1_dtype"], True))
)

def rule_24_func(arg1, arg2, solver=None, neg=False):
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
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 24
        rule_24(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_24(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype']}, neg)
