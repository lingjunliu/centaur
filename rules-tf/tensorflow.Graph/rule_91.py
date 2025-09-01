import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Check if tensor shape at some dimension is equal to some constant value and valid dimension, with positive shape. and dtype should be inside specific limit and positive dimension, limited shape values. (Rule 91)

rule_91 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(v["arg2_value"] >= 0, v["arg2_value"] < v["arg1_ndim"]), Select(v["arg1_shape"], v["arg2_value"]) == v["arg3_value"]), Select(v["arg1_shape"], 0) > 0), v["arg1_dtype"] < 12), v["arg1_ndim"] > 0), Select(v["arg1_shape"], 0) < 100)) if n else
          And(And(And(And(And(And(v["arg2_value"] >= 0, v["arg2_value"] < v["arg1_ndim"]), Select(v["arg1_shape"], v["arg2_value"]) == v["arg3_value"]), Select(v["arg1_shape"], 0) > 0), v["arg1_dtype"] < 12), v["arg1_ndim"] > 0), Select(v["arg1_shape"], 0) < 100))
)

def rule_91_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 91
        rule_91(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_91(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
