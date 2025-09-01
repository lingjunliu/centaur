import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If data's dtype is in [int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32, float64], then the maximum segment_id should be less than or equal to data's first dimension - 1 (Rule 26)

rule_26 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 5), v["arg1_dtype"] == 6), v["arg1_dtype"] == 4), v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10), v["arg1_dtype"] == 12), v["arg1_dtype"] == 13), Select(v["arg2_range"], 1) <= Select(v["arg1_shape"], 0) - 1, True)) if n else
          If(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 5), v["arg1_dtype"] == 6), v["arg1_dtype"] == 4), v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10), v["arg1_dtype"] == 12), v["arg1_dtype"] == 13), Select(v["arg2_range"], 1) <= Select(v["arg1_shape"], 0) - 1, True))
)

def rule_26_func(arg1, arg2, solver=None, neg=False):
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
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 26
        rule_26(solver, {'arg1_shape': arg1_shape, 'arg1_dtype': arg1_dtype, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_26(solver, {'arg1_shape': arg1['shape'], 'arg1_dtype': arg1['dtype'], 'arg2_range': arg2['range']}, neg)
