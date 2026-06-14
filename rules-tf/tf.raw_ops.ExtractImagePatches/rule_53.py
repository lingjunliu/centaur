import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If images is of type uint16, uint32, uint64, complex64, complex128 then images.shape[1] * images.shape[2] * images.shape[3] must be smaller than 2^16 (Rule 53)

rule_53 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(Or(Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10), v["arg1_dtype"] == 11), v["arg1_dtype"] == 12), v["arg1_dtype"] == 13), Select(v["arg1_shape"], 1) * Select(v["arg1_shape"], 2) * Select(v["arg1_shape"], 3) < 65536, True)) if n else
          If(Or(Or(Or(Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10), v["arg1_dtype"] == 11), v["arg1_dtype"] == 12), v["arg1_dtype"] == 13), Select(v["arg1_shape"], 1) * Select(v["arg1_shape"], 2) * Select(v["arg1_shape"], 3) < 65536, True))
)

def rule_53_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 53
        rule_53(solver, {'arg1_shape': arg1_shape, 'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_53(solver, {'arg1_shape': arg1['shape'], 'arg1_dtype': arg1['dtype']}, neg)
