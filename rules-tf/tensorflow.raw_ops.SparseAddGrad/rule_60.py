import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# backprop value grad can only be of certain types if all the indices have the same shape. (Rule 60)

rule_60 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And(Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0), Select(v["arg2_shape"], 0) == Select(v["arg3_shape"], 0)), Select(v["arg1_shape"], 1) == Select(v["arg2_shape"], 1)), Select(v["arg2_shape"], 1) == Select(v["arg3_shape"], 1)), Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg4_dtype"] == 7, v["arg4_dtype"] == 8), v["arg4_dtype"] == 3), v["arg4_dtype"] == 5), v["arg4_dtype"] == 2), v["arg4_dtype"] == 1), v["arg4_dtype"] == 9), v["arg4_dtype"] == 4), v["arg4_dtype"] == 6), v["arg4_dtype"] == 10), False)) if n else
          If(And(And(And(Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0), Select(v["arg2_shape"], 0) == Select(v["arg3_shape"], 0)), Select(v["arg1_shape"], 1) == Select(v["arg2_shape"], 1)), Select(v["arg2_shape"], 1) == Select(v["arg3_shape"], 1)), Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg4_dtype"] == 7, v["arg4_dtype"] == 8), v["arg4_dtype"] == 3), v["arg4_dtype"] == 5), v["arg4_dtype"] == 2), v["arg4_dtype"] == 1), v["arg4_dtype"] == 9), v["arg4_dtype"] == 4), v["arg4_dtype"] == 6), v["arg4_dtype"] == 10), False))
)

def rule_60_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_dtype = Int('arg4_dtype')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))

        # Constraints for rule 60
        rule_60(solver, {'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg3_shape': arg3_shape, 'arg4_dtype': arg4_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_60(solver, {'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg3_shape': arg3['shape'], 'arg4_dtype': arg4['dtype']}, neg)
