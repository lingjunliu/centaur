import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# var, accum, linear and grad must have the same dtype and shape (Rule 53)

rule_53 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg1_dtype"] == v["arg2_dtype"], v["arg1_dtype"] == v["arg3_dtype"]), v["arg1_dtype"] == v["arg4_dtype"]), And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(Select(v["arg1_shape"], i) == Select(v["arg2_shape"], i), And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(Select(v["arg1_shape"], i) == Select(v["arg3_shape"], i), And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) == Select(v["arg4_shape"], i)) for i in range(6)]))) for i in range(6)]))) for i in range(6)]))) if n else
          And(And(And(v["arg1_dtype"] == v["arg2_dtype"], v["arg1_dtype"] == v["arg3_dtype"]), v["arg1_dtype"] == v["arg4_dtype"]), And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(Select(v["arg1_shape"], i) == Select(v["arg2_shape"], i), And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(Select(v["arg1_shape"], i) == Select(v["arg3_shape"], i), And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) == Select(v["arg4_shape"], i)) for i in range(6)]))) for i in range(6)]))) for i in range(6)])))
)

def rule_53_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())
        arg4_dtype = Int('arg4_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))

        # Constraints for rule 53
        rule_53(solver, {'arg1_shape': arg1_shape, 'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype, 'arg2_shape': arg2_shape, 'arg3_dtype': arg3_dtype, 'arg3_shape': arg3_shape, 'arg4_dtype': arg4_dtype, 'arg4_shape': arg4_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_53(solver, {'arg1_shape': arg1['shape'], 'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_shape': arg2['shape'], 'arg3_dtype': arg3['dtype'], 'arg3_shape': arg3['shape'], 'arg4_dtype': arg4['dtype'], 'arg4_shape': arg4['shape']}, neg)
