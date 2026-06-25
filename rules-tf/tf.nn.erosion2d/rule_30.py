import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Complete validation of all input parameters (Rule 30)

rule_30 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(And(And(And(And(And(And(v["arg1_ndim"] == 4, v["arg2_ndim"] == 3), v["arg1_dtype"] == v["arg2_dtype"]), Select(v["arg1_shape"], 3) == Select(v["arg2_shape"], 2)), Select(v["arg2_shape"], 0) > 0), Select(v["arg2_shape"], 1) > 0), v["arg3_length"] == 4), Select(v["arg3_values"], 0) == 1), Select(v["arg3_values"], 3) == 1), (And([Implies(i < (v["arg3_length"] - 1 + 1), Select(v["arg3_values"], i) > 0) for i in range(6)]))), (Or(v["arg4_value"] == 28, v["arg4_value"] == 29))), v["arg5_length"] == 4), Select(v["arg5_values"], 0) == 1), Select(v["arg5_values"], 3) == 1), (And([Implies(i < (v["arg5_length"] - 1 + 1), Select(v["arg5_values"], i) > 0) for i in range(6)])))) if n else
          And(And(And(And(And(And(And(And(And(And(And(And(And(And(v["arg1_ndim"] == 4, v["arg2_ndim"] == 3), v["arg1_dtype"] == v["arg2_dtype"]), Select(v["arg1_shape"], 3) == Select(v["arg2_shape"], 2)), Select(v["arg2_shape"], 0) > 0), Select(v["arg2_shape"], 1) > 0), v["arg3_length"] == 4), Select(v["arg3_values"], 0) == 1), Select(v["arg3_values"], 3) == 1), (And([Implies(i < (v["arg3_length"] - 1 + 1), Select(v["arg3_values"], i) > 0) for i in range(6)]))), (Or(v["arg4_value"] == 28, v["arg4_value"] == 29))), v["arg5_length"] == 4), Select(v["arg5_values"], 0) == 1), Select(v["arg5_values"], 3) == 1), (And([Implies(i < (v["arg5_length"] - 1 + 1), Select(v["arg5_values"], i) > 0) for i in range(6)]))))
)

def rule_30_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
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
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False
        if not isinstance(arg4, str):
            return False
        if not (isinstance(arg5, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg5)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())
        arg4_value = Int('arg4_value')
        arg5_length = Int('arg5_length')
        arg5_values = Array('arg5_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])
        solver.add(arg4_value == list_of_string_values_tf.index(arg4))
        solver.add(arg5_length == len(arg5))
        for i in range(len(arg5)):
            arg5_values = Store(arg5_values, i, arg5[i])

        # Constraints for rule 30
        rule_30(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim, 'arg3_length': arg3_length, 'arg3_values': arg3_values, 'arg4_value': arg4_value, 'arg5_length': arg5_length, 'arg5_values': arg5_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_30(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim'], 'arg3_length': arg3['length'], 'arg3_values': arg3['values'], 'arg4_value': arg4['value'], 'arg5_length': arg5['length'], 'arg5_values': arg5['values']}, neg)
