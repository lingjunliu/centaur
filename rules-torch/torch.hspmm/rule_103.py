import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If a pre-allocated output tensor is given, it has to be of appropriate floating-point type and have dimensions to be consistent with the result (Rule 103)

rule_103 = lambda s, v, n=False: (
    s.add(Not(If((v["arg3_dtype"] != 0), And(And(And((Or(Or((v["arg3_dtype"] == 6), (v["arg3_dtype"] == 7)), (v["arg3_dtype"] == 8))), (Select(v["arg3_shape"], 0) == Select(v["arg1_shape"], 0))), (Select(v["arg3_shape"], 1) == Select(v["arg2_shape"], 1))), (v["arg3_ndim"] == 2)), True)) if n else
          If((v["arg3_dtype"] != 0), And(And(And((Or(Or((v["arg3_dtype"] == 6), (v["arg3_dtype"] == 7)), (v["arg3_dtype"] == 8))), (Select(v["arg3_shape"], 0) == Select(v["arg1_shape"], 0))), (Select(v["arg3_shape"], 1) == Select(v["arg2_shape"], 1))), (v["arg3_ndim"] == 2)), True))
)

def rule_103_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 103
        rule_103(solver, {'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg3_dtype': arg3_dtype, 'arg3_shape': arg3_shape, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_103(solver, {'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg3_dtype': arg3['dtype'], 'arg3_shape': arg3['shape'], 'arg3_ndim': arg3['ndim']}, neg)
