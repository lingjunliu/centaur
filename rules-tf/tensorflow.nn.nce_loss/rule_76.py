import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The dtype of labels must be int64 and should have a rank of 2. Also num_true should match the second dimension of labels. Number of true classes must be positive. Batch size should be greater than zero and labels, inputs should not be empty. Also labels and inputs should be non-empty tensors and they must have a shape consistent with weights. (Rule 76)

rule_76 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(And(And(v["arg1_dtype"] == 4, v["arg1_ndim"] == 2), Select(v["arg1_shape"], 1) == v["arg2_value"]), v["arg2_value"] > 0), Select(v["arg3_shape"], 0) > 0), (And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)]))), (And([Implies(i < (v["arg3_ndim"] - 1 + 1), Select(v["arg3_shape"], i) > 0) for i in range(6)]))), (Or(Or(v["arg4_dtype"] == 6, v["arg4_dtype"] == 7), v["arg4_dtype"] == 8))), v["arg4_dtype"] == v["arg3_dtype"]), v["arg4_dtype"] == v["arg5_dtype"]), Select(v["arg4_shape"], 1) == Select(v["arg3_shape"], 1))) if n else
          And(And(And(And(And(And(And(And(And(And(v["arg1_dtype"] == 4, v["arg1_ndim"] == 2), Select(v["arg1_shape"], 1) == v["arg2_value"]), v["arg2_value"] > 0), Select(v["arg3_shape"], 0) > 0), (And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)]))), (And([Implies(i < (v["arg3_ndim"] - 1 + 1), Select(v["arg3_shape"], i) > 0) for i in range(6)]))), (Or(Or(v["arg4_dtype"] == 6, v["arg4_dtype"] == 7), v["arg4_dtype"] == 8))), v["arg4_dtype"] == v["arg3_dtype"]), v["arg4_dtype"] == v["arg5_dtype"]), Select(v["arg4_shape"], 1) == Select(v["arg3_shape"], 1)))
)

def rule_76_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False
        if not isinstance(arg5, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())
        arg4_dtype = Int('arg4_dtype')
        arg5_dtype = Int('arg5_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))
        solver.add(arg5_dtype == list_of_available_dtypes.index(arg5.dtype))

        # Constraints for rule 76
        rule_76(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_ndim': arg3_ndim, 'arg3_dtype': arg3_dtype, 'arg3_shape': arg3_shape, 'arg4_dtype': arg4_dtype, 'arg4_shape': arg4_shape, 'arg5_dtype': arg5_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_76(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_ndim': arg3['ndim'], 'arg3_dtype': arg3['dtype'], 'arg3_shape': arg3['shape'], 'arg4_dtype': arg4['dtype'], 'arg4_shape': arg4['shape'], 'arg5_dtype': arg5['dtype']}, neg)
