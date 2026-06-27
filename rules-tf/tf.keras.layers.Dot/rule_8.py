import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If axes is a tuple of two integers, both must be valid non-zero axes for their respective tensors, and the size along these resolved axes must match (Rule 8)

rule_8 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(v["arg1_length"] == 2, Select(v["arg1_values"], 0) != 0), Select(v["arg1_values"], 1) != 0), (If(Select(v["arg1_values"], 0) > 0, Select(v["arg1_values"], 0) < v["arg2_ndim"], Select(v["arg1_values"], 0) > 0 - v["arg2_ndim"]))), (If(Select(v["arg1_values"], 1) > 0, Select(v["arg1_values"], 1) < v["arg3_ndim"], Select(v["arg1_values"], 1) > 0 - v["arg3_ndim"]))), Select(v["arg2_shape"], If(Select(v["arg1_values"], 0) > 0, Select(v["arg1_values"], 0), v["arg2_ndim"] + Select(v["arg1_values"], 0))) == Select(v["arg3_shape"], If(Select(v["arg1_values"], 1) > 0, Select(v["arg1_values"], 1), v["arg3_ndim"] + Select(v["arg1_values"], 1))))) if n else
          And(And(And(And(And(v["arg1_length"] == 2, Select(v["arg1_values"], 0) != 0), Select(v["arg1_values"], 1) != 0), (If(Select(v["arg1_values"], 0) > 0, Select(v["arg1_values"], 0) < v["arg2_ndim"], Select(v["arg1_values"], 0) > 0 - v["arg2_ndim"]))), (If(Select(v["arg1_values"], 1) > 0, Select(v["arg1_values"], 1) < v["arg3_ndim"], Select(v["arg1_values"], 1) > 0 - v["arg3_ndim"]))), Select(v["arg2_shape"], If(Select(v["arg1_values"], 0) > 0, Select(v["arg1_values"], 0), v["arg2_ndim"] + Select(v["arg1_values"], 0))) == Select(v["arg3_shape"], If(Select(v["arg1_values"], 1) > 0, Select(v["arg1_values"], 1), v["arg3_ndim"] + Select(v["arg1_values"], 1)))))
)

def rule_8_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 8
        rule_8(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape, 'arg3_ndim': arg3_ndim, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_8(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'], 'arg3_ndim': arg3['ndim'], 'arg3_shape': arg3['shape']}, neg)
