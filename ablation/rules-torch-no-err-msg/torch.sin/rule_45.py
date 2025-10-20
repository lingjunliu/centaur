import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the input tensor contains only zero values, and out is specified output tensor must also be of floating or complex type (Rule 45)

rule_45 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_dtype"] != 11, And((Or((Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) == 0) for i in range(6)])), (And(Select(v["arg1_range"], 0) == 0, Select(v["arg1_range"], 1) == 0)))), (Or(Or(Or(Or(v["arg2_dtype"] == 6, v["arg2_dtype"] == 7), v["arg2_dtype"] == 8), v["arg2_dtype"] == 9), v["arg2_dtype"] == 10))), True)) if n else
          If(v["arg2_dtype"] != 11, And((Or((Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) == 0) for i in range(6)])), (And(Select(v["arg1_range"], 0) == 0, Select(v["arg1_range"], 1) == 0)))), (Or(Or(Or(Or(v["arg2_dtype"] == 6, v["arg2_dtype"] == 7), v["arg2_dtype"] == 8), v["arg2_dtype"] == 9), v["arg2_dtype"] == 10))), True))
)

def rule_45_func(arg1, arg2, solver=None, neg=False):
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
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 45
        rule_45(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_45(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_dtype': arg2['dtype']}, neg)
