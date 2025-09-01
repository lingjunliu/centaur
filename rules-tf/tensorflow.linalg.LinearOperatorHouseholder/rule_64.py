import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Combined constraints and boolean checks without name (Rule 64)

rule_64 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And((Or(Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10)), v["arg1_ndim"] == 1), Select(v["arg1_shape"], 0) > 0), v["arg2_value"] == False), v["arg3_value"] == True), (Or(v["arg4_value"] == True, v["arg4_value"] == False))), (Or(v["arg5_value"] == True, v["arg5_value"] == False)))) if n else
          And(And(And(And(And(And((Or(Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10)), v["arg1_ndim"] == 1), Select(v["arg1_shape"], 0) > 0), v["arg2_value"] == False), v["arg3_value"] == True), (Or(v["arg4_value"] == True, v["arg4_value"] == False))), (Or(v["arg5_value"] == True, v["arg5_value"] == False))))
)

def rule_64_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, bool):
            return False
        if not isinstance(arg4, bool):
            return False
        if not isinstance(arg5, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Bool('arg2_value')
        arg3_value = Bool('arg3_value')
        arg4_value = Bool('arg4_value')
        arg5_value = Bool('arg5_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == arg4)
        solver.add(arg5_value == arg5)

        # Constraints for rule 64
        rule_64(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_64(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value']}, neg)
