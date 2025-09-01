import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The on and off values need to be castable to the indices type and have the same shape (Rule 115)

rule_115 = lambda s, v, n=False: (
    s.add(Not(If((v["arg3_dtype"] == 1), And(And((v["arg1_dtype"] == 1), (v["arg2_dtype"] == 1)), (Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0))), If((v["arg3_dtype"] == 5), And(And((v["arg1_dtype"] == 5), (v["arg2_dtype"] == 5)), (Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0))), If((v["arg3_dtype"] == 3), And(And((v["arg1_dtype"] == 3), (v["arg2_dtype"] == 3)), (Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0))), If((v["arg3_dtype"] == 4), And(And((v["arg1_dtype"] == 4), (v["arg2_dtype"] == 4)), (Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0))), False))))) if n else
          If((v["arg3_dtype"] == 1), And(And((v["arg1_dtype"] == 1), (v["arg2_dtype"] == 1)), (Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0))), If((v["arg3_dtype"] == 5), And(And((v["arg1_dtype"] == 5), (v["arg2_dtype"] == 5)), (Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0))), If((v["arg3_dtype"] == 3), And(And((v["arg1_dtype"] == 3), (v["arg2_dtype"] == 3)), (Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0))), If((v["arg3_dtype"] == 4), And(And((v["arg1_dtype"] == 4), (v["arg2_dtype"] == 4)), (Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0))), False)))))
)

def rule_115_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 115
        rule_115(solver, {'arg1_shape': arg1_shape, 'arg1_dtype': arg1_dtype, 'arg2_shape': arg2_shape, 'arg2_dtype': arg2_dtype, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_115(solver, {'arg1_shape': arg1['shape'], 'arg1_dtype': arg1['dtype'], 'arg2_shape': arg2['shape'], 'arg2_dtype': arg2['dtype'], 'arg3_dtype': arg3['dtype']}, neg)
