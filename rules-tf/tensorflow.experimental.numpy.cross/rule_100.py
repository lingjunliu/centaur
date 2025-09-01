import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Shape of first tensor at axisa must equal shape of second tensor at axisb, and axisa, axisb and axisc all must be valid. (Rule 100)

rule_100 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg3_value"] != -1, v["arg4_value"] != -1), v["arg5_value"] != -1), And(And(And(And(And(And((0 - v["arg1_ndim"]) <= v["arg3_value"], v["arg3_value"] < v["arg1_ndim"]), (0 - v["arg2_ndim"]) <= v["arg4_value"]), v["arg4_value"] < v["arg2_ndim"]), (0 - v["arg1_ndim"]) <= v["arg5_value"]), v["arg5_value"] < v["arg1_ndim"]), Select(v["arg1_shape"], v["arg3_value"]) == Select(v["arg2_shape"], v["arg4_value"])), True)) if n else
          If(And(And(v["arg3_value"] != -1, v["arg4_value"] != -1), v["arg5_value"] != -1), And(And(And(And(And(And((0 - v["arg1_ndim"]) <= v["arg3_value"], v["arg3_value"] < v["arg1_ndim"]), (0 - v["arg2_ndim"]) <= v["arg4_value"]), v["arg4_value"] < v["arg2_ndim"]), (0 - v["arg1_ndim"]) <= v["arg5_value"]), v["arg5_value"] < v["arg1_ndim"]), Select(v["arg1_shape"], v["arg3_value"]) == Select(v["arg2_shape"], v["arg4_value"])), True))
)

def rule_100_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
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
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False
        if not (isinstance(arg5, (int, np.integer)) and not isinstance(arg5, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')
        arg5_value = Int('arg5_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == int(arg4))
        solver.add(arg5_value == int(arg5))

        # Constraints for rule 100
        rule_100(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_100(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value']}, neg)
