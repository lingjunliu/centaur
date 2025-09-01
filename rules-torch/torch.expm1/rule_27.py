import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If out is specified, the output tensor's dtype must be compatible with the input tensor to avoid casting, only if out is not None (Rule 27)

rule_27 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg2_shape"], 0) == Select(v["arg2_shape"], 0), Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == v["arg2_dtype"], (And(v["arg1_dtype"] == 6, v["arg2_dtype"] == 7))), (And(v["arg1_dtype"] == 6, v["arg2_dtype"] == 8))), (And(v["arg1_dtype"] == 7, v["arg2_dtype"] == 6))), (And(v["arg1_dtype"] == 7, v["arg2_dtype"] == 8))), (And(v["arg1_dtype"] == 8, v["arg2_dtype"] == 6))), (And(v["arg1_dtype"] == 8, v["arg2_dtype"] == 7))), True)) if n else
          If(Select(v["arg2_shape"], 0) == Select(v["arg2_shape"], 0), Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == v["arg2_dtype"], (And(v["arg1_dtype"] == 6, v["arg2_dtype"] == 7))), (And(v["arg1_dtype"] == 6, v["arg2_dtype"] == 8))), (And(v["arg1_dtype"] == 7, v["arg2_dtype"] == 6))), (And(v["arg1_dtype"] == 7, v["arg2_dtype"] == 8))), (And(v["arg1_dtype"] == 8, v["arg2_dtype"] == 6))), (And(v["arg1_dtype"] == 8, v["arg2_dtype"] == 7))), True))
)

def rule_27_func(arg1, arg2, solver=None, neg=False):
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
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 27
        rule_27(solver, {'arg1_dtype': arg1_dtype, 'arg2_shape': arg2_shape, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_27(solver, {'arg1_dtype': arg1['dtype'], 'arg2_shape': arg2['shape'], 'arg2_dtype': arg2['dtype']}, neg)
