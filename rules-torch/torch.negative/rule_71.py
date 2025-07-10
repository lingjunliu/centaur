import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# When an out tensor is provided, the input and out tensors must have compatible dtypes for negation. Specifically, the out tensor's dtype must be greater or equal to the input tensor's dtype. Additionally, if input is float32, then out cannot be float. (Rule 71)

rule_71 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_ndim"] > 0, (If(And(v["arg1_dtype"] >= 1, v["arg1_dtype"] <= 5), v["arg2_dtype"] >= v["arg1_dtype"], If(And(v["arg1_dtype"] >= 6, v["arg1_dtype"] <= 8), And((v["arg2_dtype"] >= v["arg1_dtype"]), If(v["arg1_dtype"] == 7, v["arg2_dtype"] != 8, False)), False))), False)) if n else
          If(v["arg2_ndim"] > 0, (If(And(v["arg1_dtype"] >= 1, v["arg1_dtype"] <= 5), v["arg2_dtype"] >= v["arg1_dtype"], If(And(v["arg1_dtype"] >= 6, v["arg1_dtype"] <= 8), And((v["arg2_dtype"] >= v["arg1_dtype"]), If(v["arg1_dtype"] == 7, v["arg2_dtype"] != 8, False)), False))), False))
)

def rule_71_func(arg1, arg2, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 71
        rule_71(solver, {'arg1_dtype': arg1_dtype, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_71(solver, {'arg1_dtype': arg1['dtype'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype']}, neg)
