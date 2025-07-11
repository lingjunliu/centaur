import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the input tensor has more than 1 dimension, and not more than 5, and is not a tuple and has floating point type, and has the correct range, and last dimension is 3, and the ouput tensor shape is same as input, and dtypes are same and is not complex and min <= max and its checked element wise and if its 3d dimensions are positive, the conditions are satified (Rule 35)

rule_35 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_ndim"] > 1, v["arg1_ndim"] <= 5), (And(And(And(And(And(And((Or(Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 9)), (And(Select(v["arg1_range"], 0) >= 0, Select(v["arg1_range"], 1) <= 1))), (Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 3)), (And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) == Select(v["arg2_shape"], i)) for i in range(6)]))), (v["arg1_dtype"] == v["arg2_dtype"])), (Select(v["arg1_range"], 0) <= Select(v["arg1_range"], 1))), If(v["arg1_ndim"] == 3, (And(And(Select(v["arg1_shape"], 0) > 0, Select(v["arg1_shape"], 1) > 0), Select(v["arg1_shape"], 2) > 0)), False))), False)) if n else
          If(And(v["arg1_ndim"] > 1, v["arg1_ndim"] <= 5), (And(And(And(And(And(And((Or(Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 9)), (And(Select(v["arg1_range"], 0) >= 0, Select(v["arg1_range"], 1) <= 1))), (Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 3)), (And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) == Select(v["arg2_shape"], i)) for i in range(6)]))), (v["arg1_dtype"] == v["arg2_dtype"])), (Select(v["arg1_range"], 0) <= Select(v["arg1_range"], 1))), If(v["arg1_ndim"] == 3, (And(And(Select(v["arg1_shape"], 0) > 0, Select(v["arg1_shape"], 1) > 0), Select(v["arg1_shape"], 2) > 0)), False))), False))
)

def rule_35_func(arg1, arg2, solver=None, neg=False):
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
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 35
        rule_35(solver, {'arg1_shape': arg1_shape, 'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_35(solver, {'arg1_shape': arg1['shape'], 'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_shape': arg2['shape']}, neg)
