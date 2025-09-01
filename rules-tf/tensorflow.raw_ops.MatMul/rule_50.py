import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Shape and dtype requirements for MatMul inputs (Rule 50)

rule_50 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(v["arg1_ndim"] == 2, v["arg2_ndim"] == 2), v["arg1_dtype"] == v["arg2_dtype"]), (Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 4), v["arg1_dtype"] == 5), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10), v["arg1_dtype"] == 11), v["arg1_dtype"] == 12), v["arg1_dtype"] == 13))), (Or(Or(Or((And(And(v["arg3_value"] == False, v["arg4_value"] == False), Select(v["arg1_shape"], 1) == Select(v["arg2_shape"], 0))), (And(And(v["arg3_value"] == False, v["arg4_value"] == True), Select(v["arg1_shape"], 1) == Select(v["arg2_shape"], 1)))), (And(And(v["arg3_value"] == True, v["arg4_value"] == False), Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0)))), (And(And(v["arg3_value"] == True, v["arg4_value"] == True), Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 1)))))), (Or(v["arg5_value"] == True, v["arg5_value"] == False))), (Or(v["arg6_value"] == True, v["arg6_value"] == False)))) if n else
          And(And(And(And(And(And(v["arg1_ndim"] == 2, v["arg2_ndim"] == 2), v["arg1_dtype"] == v["arg2_dtype"]), (Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 4), v["arg1_dtype"] == 5), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10), v["arg1_dtype"] == 11), v["arg1_dtype"] == 12), v["arg1_dtype"] == 13))), (Or(Or(Or((And(And(v["arg3_value"] == False, v["arg4_value"] == False), Select(v["arg1_shape"], 1) == Select(v["arg2_shape"], 0))), (And(And(v["arg3_value"] == False, v["arg4_value"] == True), Select(v["arg1_shape"], 1) == Select(v["arg2_shape"], 1)))), (And(And(v["arg3_value"] == True, v["arg4_value"] == False), Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0)))), (And(And(v["arg3_value"] == True, v["arg4_value"] == True), Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 1)))))), (Or(v["arg5_value"] == True, v["arg5_value"] == False))), (Or(v["arg6_value"] == True, v["arg6_value"] == False))))
)

def rule_50_func(arg1, arg2, arg3, arg4, arg5, arg6, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, bool):
            return False
        if not isinstance(arg4, bool):
            return False
        if not isinstance(arg5, bool):
            return False
        if not isinstance(arg6, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg3_value = Bool('arg3_value')
        arg4_value = Bool('arg4_value')
        arg5_value = Bool('arg5_value')
        arg6_value = Bool('arg6_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == arg4)
        solver.add(arg5_value == arg5)
        solver.add(arg6_value == arg6)

        # Constraints for rule 50
        rule_50(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value, 'arg6_value': arg6_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_50(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value'], 'arg6_value': arg6['value']}, neg)
