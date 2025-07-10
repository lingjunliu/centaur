import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# the first dimension size of the input tensor should be power of 2 if GPU Arch is SM53 or greater (Rule 81)

rule_81 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_shape"], 0) == 0, True, (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_shape"], 0) == 2, Select(v["arg1_shape"], 0) == 4), Select(v["arg1_shape"], 0) == 8), Select(v["arg1_shape"], 0) == 16), Select(v["arg1_shape"], 0) == 32), Select(v["arg1_shape"], 0) == 64), Select(v["arg1_shape"], 0) == 128), Select(v["arg1_shape"], 0) == 256), Select(v["arg1_shape"], 0) == 512), Select(v["arg1_shape"], 0) == 1024), Select(v["arg1_shape"], 0) == 2048)))) if n else
          If(Select(v["arg1_shape"], 0) == 0, True, (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_shape"], 0) == 2, Select(v["arg1_shape"], 0) == 4), Select(v["arg1_shape"], 0) == 8), Select(v["arg1_shape"], 0) == 16), Select(v["arg1_shape"], 0) == 32), Select(v["arg1_shape"], 0) == 64), Select(v["arg1_shape"], 0) == 128), Select(v["arg1_shape"], 0) == 256), Select(v["arg1_shape"], 0) == 512), Select(v["arg1_shape"], 0) == 1024), Select(v["arg1_shape"], 0) == 2048))))
)

def rule_81_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])

        # Constraints for rule 81
        rule_81(solver, {'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_81(solver, {'arg1_shape': arg1['shape']}, neg)
