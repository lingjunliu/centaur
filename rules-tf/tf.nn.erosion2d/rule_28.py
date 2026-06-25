import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Combined check for data_format, value, filters, and their dimensions/types (Rule 28)

rule_28 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(And(And(And(And(v["arg1_value"] == 33, v["arg2_ndim"] == 4), v["arg3_ndim"] == 3), v["arg2_dtype"] == v["arg3_dtype"]), Select(v["arg2_shape"], 3) == Select(v["arg3_shape"], 2)), v["arg4_length"] == 4), Select(v["arg4_values"], 0) == 1), Select(v["arg4_values"], 3) == 1), (And([Implies(i < (v["arg4_length"] - 1 + 1), Select(v["arg4_values"], i) > 0) for i in range(6)]))), v["arg5_length"] == 4), Select(v["arg5_values"], 0) == 1), Select(v["arg5_values"], 3) == 1), (And([Implies(i < (v["arg5_length"] - 1 + 1), Select(v["arg5_values"], i) > 0) for i in range(6)])))) if n else
          And(And(And(And(And(And(And(And(And(And(And(And(v["arg1_value"] == 33, v["arg2_ndim"] == 4), v["arg3_ndim"] == 3), v["arg2_dtype"] == v["arg3_dtype"]), Select(v["arg2_shape"], 3) == Select(v["arg3_shape"], 2)), v["arg4_length"] == 4), Select(v["arg4_values"], 0) == 1), Select(v["arg4_values"], 3) == 1), (And([Implies(i < (v["arg4_length"] - 1 + 1), Select(v["arg4_values"], i) > 0) for i in range(6)]))), v["arg5_length"] == 4), Select(v["arg5_values"], 0) == 1), Select(v["arg5_values"], 3) == 1), (And([Implies(i < (v["arg5_length"] - 1 + 1), Select(v["arg5_values"], i) > 0) for i in range(6)]))))
)

def rule_28_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not (isinstance(arg4, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False
        if not (isinstance(arg5, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg5)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')
        arg4_length = Int('arg4_length')
        arg4_values = Array('arg4_values', IntSort(), IntSort())
        arg5_length = Int('arg5_length')
        arg5_values = Array('arg5_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_length == len(arg4))
        for i in range(len(arg4)):
            arg4_values = Store(arg4_values, i, arg4[i])
        solver.add(arg5_length == len(arg5))
        for i in range(len(arg5)):
            arg5_values = Store(arg5_values, i, arg5[i])

        # Constraints for rule 28
        rule_28(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim, 'arg3_dtype': arg3_dtype, 'arg3_shape': arg3_shape, 'arg3_ndim': arg3_ndim, 'arg4_length': arg4_length, 'arg4_values': arg4_values, 'arg5_length': arg5_length, 'arg5_values': arg5_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_28(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim'], 'arg3_dtype': arg3['dtype'], 'arg3_shape': arg3['shape'], 'arg3_ndim': arg3['ndim'], 'arg4_length': arg4['length'], 'arg4_values': arg4['values'], 'arg5_length': arg5['length'], 'arg5_values': arg5['values']}, neg)
