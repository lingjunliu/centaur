import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Focus on preventing dimensions incompatibilities based on previous two tensor information focusing on dense shape. (Rule 100)

rule_100 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_dtype"] != 0, v["arg1_ndim"] == 1, Select(v["arg2_shape"], 1) == v["arg1_ndim"])) if n else
          If(v["arg2_dtype"] != 0, v["arg1_ndim"] == 1, Select(v["arg2_shape"], 1) == v["arg1_ndim"]))
)

def rule_100_func(arg1, arg2, solver=None, neg=False):
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
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 100
        rule_100(solver, {'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_100(solver, {'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_shape': arg2['shape']}, neg)
