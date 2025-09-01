import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Shape of paddings is valid with respect to the input tensor's ndim and the constant_values tensor (Rule 49)

rule_49 = lambda s, v, n=False: (
    s.add(Not(And(And((v["arg2_ndim"] == 2), (Or(Or(Or(Or((And(v["arg1_ndim"] == 0, Select(v["arg2_shape"], 0) == 0)), (And(v["arg1_ndim"] == 1, Select(v["arg2_shape"], 0) == 1))), (And(v["arg1_ndim"] == 2, Select(v["arg2_shape"], 0) == 2))), (And(v["arg1_ndim"] == 3, Select(v["arg2_shape"], 0) == 3))), (And(v["arg1_ndim"] == 4, Select(v["arg2_shape"], 0) == 4))))), v["arg1_dtype"] == v["arg3_dtype"])) if n else
          And(And((v["arg2_ndim"] == 2), (Or(Or(Or(Or((And(v["arg1_ndim"] == 0, Select(v["arg2_shape"], 0) == 0)), (And(v["arg1_ndim"] == 1, Select(v["arg2_shape"], 0) == 1))), (And(v["arg1_ndim"] == 2, Select(v["arg2_shape"], 0) == 2))), (And(v["arg1_ndim"] == 3, Select(v["arg2_shape"], 0) == 3))), (And(v["arg1_ndim"] == 4, Select(v["arg2_shape"], 0) == 4))))), v["arg1_dtype"] == v["arg3_dtype"]))
)

def rule_49_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 49
        rule_49(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_49(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim'], 'arg3_dtype': arg3['dtype']}, neg)
