import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The image size channel must be less than or equal to the max value for the image_size's dtype (Rule 104)

rule_104 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 5, Select(v["arg1_shape"], 2) <= 255, If(v["arg1_dtype"] == 1, Select(v["arg1_shape"], 2) <= 127, If(v["arg1_dtype"] == 2, Select(v["arg1_shape"], 2) <= 32767, If(v["arg1_dtype"] == 3, Select(v["arg1_shape"], 2) <= 2147483647, If(v["arg1_dtype"] == 4, Select(v["arg1_shape"], 2) <= 9223372036854775807, True)))))) if n else
          If(v["arg1_dtype"] == 5, Select(v["arg1_shape"], 2) <= 255, If(v["arg1_dtype"] == 1, Select(v["arg1_shape"], 2) <= 127, If(v["arg1_dtype"] == 2, Select(v["arg1_shape"], 2) <= 32767, If(v["arg1_dtype"] == 3, Select(v["arg1_shape"], 2) <= 2147483647, If(v["arg1_dtype"] == 4, Select(v["arg1_shape"], 2) <= 9223372036854775807, True))))))
)

def rule_104_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 104
        rule_104(solver, {'arg1_shape': arg1_shape, 'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_104(solver, {'arg1_shape': arg1['shape'], 'arg1_dtype': arg1['dtype']}, neg)
