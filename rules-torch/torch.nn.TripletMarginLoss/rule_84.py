import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# All conditions met: Matching shape, dimensions, floating dtype, positive margin, non-empty, consistent tensor size. (Rule 84)

rule_84 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And((And(v["arg1_ndim"] == v["arg2_ndim"], v["arg1_ndim"] == v["arg3_ndim"])), (And([Implies(i < (If(v["arg1_ndim"] > 0, v["arg1_ndim"] - 1, 0) + 1), (And(Select(v["arg1_shape"], i) == Select(v["arg2_shape"], i), Select(v["arg1_shape"], i) == Select(v["arg3_shape"], i)))) for i in range(6)]))), (Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 6))), (v["arg4_value"] > 0)), (And([Implies(j < (If(v["arg1_ndim"] > 0, v["arg1_ndim"] - 1, 0) + 1), Select(v["arg1_shape"], j) > 0) for j in range(6)]))), Select(v["arg1_shape"], 0) > 0)) if n else
          And(And(And(And(And((And(v["arg1_ndim"] == v["arg2_ndim"], v["arg1_ndim"] == v["arg3_ndim"])), (And([Implies(i < (If(v["arg1_ndim"] > 0, v["arg1_ndim"] - 1, 0) + 1), (And(Select(v["arg1_shape"], i) == Select(v["arg2_shape"], i), Select(v["arg1_shape"], i) == Select(v["arg3_shape"], i)))) for i in range(6)]))), (Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 6))), (v["arg4_value"] > 0)), (And([Implies(j < (If(v["arg1_ndim"] > 0, v["arg1_ndim"] - 1, 0) + 1), Select(v["arg1_shape"], j) > 0) for j in range(6)]))), Select(v["arg1_shape"], 0) > 0))
)

def rule_84_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        if not isinstance(arg4, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_value = Real('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg4_value == arg4)

        # Constraints for rule 84
        rule_84(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape, 'arg3_ndim': arg3_ndim, 'arg3_shape': arg3_shape, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_84(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'], 'arg3_ndim': arg3['ndim'], 'arg3_shape': arg3['shape'], 'arg4_value': arg4['value']}, neg)
