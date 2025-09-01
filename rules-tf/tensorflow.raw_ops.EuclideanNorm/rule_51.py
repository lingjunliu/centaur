import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If input is int8 then output should also be a float32 or float64 if keepdims is False and all dimension are being reduced. Otherwise it should be int8 (Rule 51)

rule_51 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 1, If(And(Select(v["arg3_shape"], 0) == v["arg1_ndim"], v["arg4_value"] == False), Or((v["arg2_dtype"] == 7), (v["arg2_dtype"] == 8)), v["arg2_dtype"] == 1), True)) if n else
          If(v["arg1_dtype"] == 1, If(And(Select(v["arg3_shape"], 0) == v["arg1_ndim"], v["arg4_value"] == False), Or((v["arg2_dtype"] == 7), (v["arg2_dtype"] == 8)), v["arg2_dtype"] == 1), True))
)

def rule_51_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        if not isinstance(arg4, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_value = Bool('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg4_value == arg4)

        # Constraints for rule 51
        rule_51(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg3_shape': arg3_shape, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_51(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg3_shape': arg3['shape'], 'arg4_value': arg4['value']}, neg)
