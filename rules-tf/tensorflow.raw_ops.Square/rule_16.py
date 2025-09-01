import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The size of the tensor in bytes should be less than a limit (Rule 16)

rule_16 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 0, Select(v["arg1_shape"], 0) * 1 < 2000000000, If(v["arg1_dtype"] == 1, Select(v["arg1_shape"], 0) * 1 < 2000000000, If(v["arg1_dtype"] == 2, Select(v["arg1_shape"], 0) * 2 < 2000000000, If(v["arg1_dtype"] == 3, Select(v["arg1_shape"], 0) * 4 < 2000000000, If(v["arg1_dtype"] == 4, Select(v["arg1_shape"], 0) * 8 < 2000000000, If(v["arg1_dtype"] == 5, Select(v["arg1_shape"], 0) * 1 < 2000000000, If(v["arg1_dtype"] == 6, Select(v["arg1_shape"], 0) * 2 < 2000000000, If(v["arg1_dtype"] == 7, Select(v["arg1_shape"], 0) * 4 < 2000000000, If(v["arg1_dtype"] == 8, Select(v["arg1_shape"], 0) * 8 < 2000000000, True)))))))))) if n else
          If(v["arg1_dtype"] == 0, Select(v["arg1_shape"], 0) * 1 < 2000000000, If(v["arg1_dtype"] == 1, Select(v["arg1_shape"], 0) * 1 < 2000000000, If(v["arg1_dtype"] == 2, Select(v["arg1_shape"], 0) * 2 < 2000000000, If(v["arg1_dtype"] == 3, Select(v["arg1_shape"], 0) * 4 < 2000000000, If(v["arg1_dtype"] == 4, Select(v["arg1_shape"], 0) * 8 < 2000000000, If(v["arg1_dtype"] == 5, Select(v["arg1_shape"], 0) * 1 < 2000000000, If(v["arg1_dtype"] == 6, Select(v["arg1_shape"], 0) * 2 < 2000000000, If(v["arg1_dtype"] == 7, Select(v["arg1_shape"], 0) * 4 < 2000000000, If(v["arg1_dtype"] == 8, Select(v["arg1_shape"], 0) * 8 < 2000000000, True))))))))))
)

def rule_16_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 16
        rule_16(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_16(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape']}, neg)
