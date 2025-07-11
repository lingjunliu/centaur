import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if adj_x is false and adj_y is true, and number of dimension is greater than 1, then the output last two dimension are determined by the last two dimensions of the input (Rule 65)

rule_65 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And(And(v["arg4_value"] == False, v["arg5_value"] == True), v["arg1_ndim"] > 1), v["arg2_ndim"] > 1), v["arg3_ndim"] > 1), And(Select(v["arg3_shape"], v["arg3_ndim"] - 2) == Select(v["arg1_shape"], v["arg1_ndim"] - 2), Select(v["arg3_shape"], v["arg3_ndim"] - 1) == Select(v["arg2_shape"], v["arg2_ndim"] - 2)), False)) if n else
          If(And(And(And(And(v["arg4_value"] == False, v["arg5_value"] == True), v["arg1_ndim"] > 1), v["arg2_ndim"] > 1), v["arg3_ndim"] > 1), And(Select(v["arg3_shape"], v["arg3_ndim"] - 2) == Select(v["arg1_shape"], v["arg1_ndim"] - 2), Select(v["arg3_shape"], v["arg3_ndim"] - 1) == Select(v["arg2_shape"], v["arg2_ndim"] - 2)), False))
)

def rule_65_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
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
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, bool):
            return False
        if not isinstance(arg5, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_value = Bool('arg4_value')
        arg5_value = Bool('arg5_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg4_value == arg4)
        solver.add(arg5_value == arg5)

        # Constraints for rule 65
        rule_65(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim, 'arg3_shape': arg3_shape, 'arg3_ndim': arg3_ndim, 'arg4_value': arg4_value, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_65(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim'], 'arg3_shape': arg3['shape'], 'arg3_ndim': arg3['ndim'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value']}, neg)
