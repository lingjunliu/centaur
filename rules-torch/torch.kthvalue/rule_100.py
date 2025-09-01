import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Check k is valid for a given dimension to avoid RuntimeError, if specified and dimension exist. Must be at least one element, prevent IndexError for empty dim, ndim >0, k can not be zero. Also, for scalar k ==1 and also check that shape is not zero , Check scalar tensor, and check dimension exists and is valid before accessing shape (Rule 100)

rule_100 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] > 0, If(And(v["arg3_value"] < v["arg1_ndim"], v["arg3_value"] >= (0 - v["arg1_ndim"])), If(And(Select(v["arg1_shape"], 0) > 0, Select(v["arg1_shape"], v["arg3_value"]) > 0), And(v["arg2_value"] <= Select(v["arg1_shape"], v["arg3_value"]), v["arg2_value"] > 0), If(v["arg1_ndim"] == 0, v["arg2_value"] == 1, True)), True), True)) if n else
          If(v["arg1_ndim"] > 0, If(And(v["arg3_value"] < v["arg1_ndim"], v["arg3_value"] >= (0 - v["arg1_ndim"])), If(And(Select(v["arg1_shape"], 0) > 0, Select(v["arg1_shape"], v["arg3_value"]) > 0), And(v["arg2_value"] <= Select(v["arg1_shape"], v["arg3_value"]), v["arg2_value"] > 0), If(v["arg1_ndim"] == 0, v["arg2_value"] == 1, True)), True), True))
)

def rule_100_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 100
        rule_100(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_100(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
