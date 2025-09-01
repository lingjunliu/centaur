import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# All parameters are valid (Rule 44)

rule_44 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(v["arg1_value"] == False, v["arg2_value"] == True), (Or(Or(Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8), v["arg3_dtype"] == 9), v["arg3_dtype"] == 10))), v["arg3_ndim"] == 1), Select(v["arg3_shape"], 0) > 0), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg4_value"] == 6, v["arg4_value"] == 7), v["arg4_value"] == 8), v["arg4_value"] == 9), v["arg4_value"] == 10), v["arg4_value"] == 11), v["arg4_value"] == 12), v["arg4_value"] == 13), v["arg4_value"] == 14), v["arg4_value"] == 15), v["arg4_value"] == 16), v["arg4_value"] == 17), v["arg4_value"] == 18), v["arg4_value"] == 19), v["arg4_value"] == 20), v["arg4_value"] == 21), v["arg4_value"] == 22), v["arg4_value"] == 23), v["arg4_value"] == 24), v["arg4_value"] == 25))), (Or(v["arg5_value"] == True, v["arg5_value"] == False))), (Or(v["arg6_value"] == True, v["arg6_value"] == False)))) if n else
          And(And(And(And(And(And(And(v["arg1_value"] == False, v["arg2_value"] == True), (Or(Or(Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8), v["arg3_dtype"] == 9), v["arg3_dtype"] == 10))), v["arg3_ndim"] == 1), Select(v["arg3_shape"], 0) > 0), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg4_value"] == 6, v["arg4_value"] == 7), v["arg4_value"] == 8), v["arg4_value"] == 9), v["arg4_value"] == 10), v["arg4_value"] == 11), v["arg4_value"] == 12), v["arg4_value"] == 13), v["arg4_value"] == 14), v["arg4_value"] == 15), v["arg4_value"] == 16), v["arg4_value"] == 17), v["arg4_value"] == 18), v["arg4_value"] == 19), v["arg4_value"] == 20), v["arg4_value"] == 21), v["arg4_value"] == 22), v["arg4_value"] == 23), v["arg4_value"] == 24), v["arg4_value"] == 25))), (Or(v["arg5_value"] == True, v["arg5_value"] == False))), (Or(v["arg6_value"] == True, v["arg6_value"] == False))))
)

def rule_44_func(arg1, arg2, arg3, arg4, arg5, arg6, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, str):
            return False
        if not isinstance(arg5, bool):
            return False
        if not isinstance(arg6, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')
        arg4_value = String('arg4_value')
        arg5_value = Bool('arg5_value')
        arg6_value = Bool('arg6_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_value == list_of_string_values_tf.index(arg4))
        solver.add(arg5_value == arg5)
        solver.add(arg6_value == arg6)

        # Constraints for rule 44
        rule_44(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape, 'arg3_ndim': arg3_ndim, 'arg3_dtype': arg3_dtype, 'arg4_value': arg4_value, 'arg5_value': arg5_value, 'arg6_value': arg6_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_44(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape'], 'arg3_ndim': arg3['ndim'], 'arg3_dtype': arg3['dtype'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value'], 'arg6_value': arg6['value']}, neg)
