import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Dtype should be floating point or complex, otherwise gradients are not supported (Rule 65)

rule_65 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_dtype"] == 1, False, If(v["arg2_dtype"] == 2, False, If(v["arg2_dtype"] == 3, False, If(v["arg2_dtype"] == 4, False, If(v["arg2_dtype"] == 5, False, If(v["arg2_dtype"] == 11, False, True))))))) if n else
          If(v["arg2_dtype"] == 1, False, If(v["arg2_dtype"] == 2, False, If(v["arg2_dtype"] == 3, False, If(v["arg2_dtype"] == 4, False, If(v["arg2_dtype"] == 5, False, If(v["arg2_dtype"] == 11, False, True)))))))
)

def rule_65_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 65
        rule_65(solver, {'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_65(solver, {'arg2_dtype': arg2['dtype']}, neg)
