import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# seed must be 1 dimensional with 2 elements and alpha,beta must have correct data types. (Rule 40)

rule_40 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg1_ndim"] == 1, Select(v["arg1_shape"], 0) == 2), (Or(Or((v["arg2_dtype"] == 6), (v["arg2_dtype"] == 7)), (v["arg2_dtype"] == 8)))), (Or(Or((v["arg3_dtype"] == 6), (v["arg3_dtype"] == 7)), (v["arg3_dtype"] == 8))))) if n else
          And(And(And(v["arg1_ndim"] == 1, Select(v["arg1_shape"], 0) == 2), (Or(Or((v["arg2_dtype"] == 6), (v["arg2_dtype"] == 7)), (v["arg2_dtype"] == 8)))), (Or(Or((v["arg3_dtype"] == 6), (v["arg3_dtype"] == 7)), (v["arg3_dtype"] == 8)))))
)

def rule_40_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 40
        rule_40(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_40(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype'], 'arg3_dtype': arg3['dtype']}, neg)
