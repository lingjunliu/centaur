import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If axis is not None, then repeats should be a 1D tensor of int and len(repeats (Rule 76)

rule_76 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg3_value"] >= 0, v["arg3_value"] < v["arg1_ndim"]), And(And(Select(v["arg1_shape"], v["arg3_value"]) == Select(v["arg2_shape"], 0), v["arg2_ndim"] == 1), (Or(Or(Or(v["arg2_dtype"] == 2, v["arg2_dtype"] == 3), v["arg2_dtype"] == 4), v["arg2_dtype"] == 5))), True)) if n else
          If(And(v["arg3_value"] >= 0, v["arg3_value"] < v["arg1_ndim"]), And(And(Select(v["arg1_shape"], v["arg3_value"]) == Select(v["arg2_shape"], 0), v["arg2_ndim"] == 1), (Or(Or(Or(v["arg2_dtype"] == 2, v["arg2_dtype"] == 3), v["arg2_dtype"] == 4), v["arg2_dtype"] == 5))), True))
)

def rule_76_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 76
        rule_76(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_76(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value']}, neg)
