import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If out is provided, its dtype must match input's and both must be floating or complex point type, and out's shape should be consistent with input's along the reduction dimension if keepdim is false (Rule 77)

rule_77 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg1_dtype"] == v["arg2_dtype"], 6 <= v["arg1_dtype"]), v["arg1_dtype"] <= 11), If(v["arg3_value"] == 6, True, If(v["arg4_value"] == False, Select(v["arg1_shape"], v["arg3_value"]) == Select(v["arg2_shape"], v["arg3_value"]), False)))) if n else
          And(And(And(v["arg1_dtype"] == v["arg2_dtype"], 6 <= v["arg1_dtype"]), v["arg1_dtype"] <= 11), If(v["arg3_value"] == 6, True, If(v["arg4_value"] == False, Select(v["arg1_shape"], v["arg3_value"]) == Select(v["arg2_shape"], v["arg3_value"]), False))))
)

def rule_77_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        if not ((isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)) or isinstance(arg3, str)):
            return False
        if not isinstance(arg4, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg4_value = Bool('arg4_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg4_value == arg4)

        # Constraints for rule 77
        rule_77(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg2_dtype': arg2_dtype, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_77(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg2_dtype': arg2['dtype'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
