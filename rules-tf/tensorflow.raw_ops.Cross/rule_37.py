import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if a is 3-element vector, then b must have appropriate dtype (i.e. oneOf(float32, float64, int32, uint8, int16, int8, int64, bfloat16, uint16, half, uint32, uint64 (Rule 37)

rule_37 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_ndim"] == 1, Select(v["arg1_shape"], 0) == 3), Or(Or(Or(Or(Or(Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8), v["arg2_dtype"] == 3), v["arg2_dtype"] == 6), v["arg2_dtype"] == 2), v["arg2_dtype"] == 1), v["arg2_dtype"] == 4), False)) if n else
          If(And(v["arg1_ndim"] == 1, Select(v["arg1_shape"], 0) == 3), Or(Or(Or(Or(Or(Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8), v["arg2_dtype"] == 3), v["arg2_dtype"] == 6), v["arg2_dtype"] == 2), v["arg2_dtype"] == 1), v["arg2_dtype"] == 4), False))
)

def rule_37_func(arg1, arg2, solver=None, neg=False):
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
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 37
        rule_37(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_37(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype']}, neg)
