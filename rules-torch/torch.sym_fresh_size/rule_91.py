import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# two tensors have same shape at dimension 0 and at dimension 1 and the product of the shapes is less than 1000 or different data types or min of both tensors is greater than 0  (Rule 91)

rule_91 = lambda s, v, n=False: (
    s.add(Not(Or(Or((And(And(Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0), Select(v["arg1_shape"], 1) == Select(v["arg2_shape"], 1)), Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) < 1000)), v["arg1_dtype"] != v["arg2_dtype"]), (And(Select(v["arg1_range"], 0) > 0, Select(v["arg2_range"], 0) > 0)))) if n else
          Or(Or((And(And(Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0), Select(v["arg1_shape"], 1) == Select(v["arg2_shape"], 1)), Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) < 1000)), v["arg1_dtype"] != v["arg2_dtype"]), (And(Select(v["arg1_range"], 0) > 0, Select(v["arg2_range"], 0) > 0))))
)

def rule_91_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 91
        rule_91(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range, 'arg1_shape': arg1_shape, 'arg2_dtype': arg2_dtype, 'arg2_range': arg2_range, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_91(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range'], 'arg1_shape': arg1['shape'], 'arg2_dtype': arg2['dtype'], 'arg2_range': arg2['range'], 'arg2_shape': arg2['shape']}, neg)
