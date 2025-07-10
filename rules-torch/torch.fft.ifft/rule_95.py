import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# The first dimension of input should be a power of 2 if GPU Arch is SM53 or greater, check only for complex64, complex128 and Half (Rule 95)

rule_95 = lambda s, v, n=False: (
    s.add(Not(If(And((Or(Or(v["arg1_dtype"] == 10, v["arg1_dtype"] == 11), v["arg1_dtype"] == 7)), Select(v["arg1_shape"], 0) != 0), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_shape"], 0) == 2, Select(v["arg1_shape"], 0) == 4), Select(v["arg1_shape"], 0) == 8), Select(v["arg1_shape"], 0) == 16), Select(v["arg1_shape"], 0) == 32), Select(v["arg1_shape"], 0) == 64), Select(v["arg1_shape"], 0) == 128), Select(v["arg1_shape"], 0) == 256), Select(v["arg1_shape"], 0) == 512), Select(v["arg1_shape"], 0) == 1024), Select(v["arg1_shape"], 0) == 2048)), False)) if n else
          If(And((Or(Or(v["arg1_dtype"] == 10, v["arg1_dtype"] == 11), v["arg1_dtype"] == 7)), Select(v["arg1_shape"], 0) != 0), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_shape"], 0) == 2, Select(v["arg1_shape"], 0) == 4), Select(v["arg1_shape"], 0) == 8), Select(v["arg1_shape"], 0) == 16), Select(v["arg1_shape"], 0) == 32), Select(v["arg1_shape"], 0) == 64), Select(v["arg1_shape"], 0) == 128), Select(v["arg1_shape"], 0) == 256), Select(v["arg1_shape"], 0) == 512), Select(v["arg1_shape"], 0) == 1024), Select(v["arg1_shape"], 0) == 2048)), False))
)

def rule_95_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 95
        rule_95(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_95(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape']}, neg)
