import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if range_given is true, and axis is not -1 and if tensor input_min/max then ensure the leading dimension of the inputs match (Rule 95)

rule_95 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And(v["arg3_value"], v["arg4_value"] != -1), v["arg1_ndim"] > 0), v["arg2_ndim"] > 0), And(Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0), Select(v["arg1_shape"], 0) == Select(v["arg5_shape"], v["arg4_value"])), True)) if n else
          If(And(And(And(v["arg3_value"], v["arg4_value"] != -1), v["arg1_ndim"] > 0), v["arg2_ndim"] > 0), And(Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0), Select(v["arg1_shape"], 0) == Select(v["arg5_shape"], v["arg4_value"])), True))
)

def rule_95_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, bool):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False
        if not isinstance(arg5, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = Bool('arg3_value')
        arg4_value = Int('arg4_value')
        arg5_shape = Array('arg5_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == int(arg4))
        for i in range(arg5.ndim):
            arg5_shape = Store(arg5_shape, i, arg5.shape[i])

        # Constraints for rule 95
        rule_95(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_shape': arg5_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_95(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_shape': arg5['shape']}, neg)
