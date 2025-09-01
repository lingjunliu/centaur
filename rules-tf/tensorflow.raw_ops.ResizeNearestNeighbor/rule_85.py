import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if the image is grayscale then number of channels should be 1 (Rule 85)

rule_85 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 5), v["arg1_dtype"] == 2), v["arg1_dtype"] == 6), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 14), True, Select(v["arg1_shape"], 3) == 1)) if n else
          If(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 5), v["arg1_dtype"] == 2), v["arg1_dtype"] == 6), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 14), True, Select(v["arg1_shape"], 3) == 1))
)

def rule_85_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 85
        rule_85(solver, {'arg1_shape': arg1_shape, 'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_85(solver, {'arg1_shape': arg1['shape'], 'arg1_dtype': arg1['dtype']}, neg)
