import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Check Indices Properties: dtype, rank >= 1, also handles the shape of the update (Rule 104)

rule_104 = lambda s, v, n=False: (
    s.add(Not(And(And((Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4)), (v["arg2_ndim"] > 0)), If(Or(Or(v["arg1_ndim"] < 1, v["arg2_ndim"] < 1), v["arg3_ndim"] < 0), False, If(v["arg3_ndim"] == 0, True, And((And([Implies(i < (If(v["arg2_ndim"] > 1, v["arg2_ndim"] - 1 - 1, 0) + 1), Select(v["arg2_shape"], i) == Select(v["arg3_shape"], i)) for i in range(6)])), (Select(v["arg2_shape"], v["arg2_ndim"] - 1) <= v["arg1_ndim"])))))) if n else
          And(And((Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4)), (v["arg2_ndim"] > 0)), If(Or(Or(v["arg1_ndim"] < 1, v["arg2_ndim"] < 1), v["arg3_ndim"] < 0), False, If(v["arg3_ndim"] == 0, True, And((And([Implies(i < (If(v["arg2_ndim"] > 1, v["arg2_ndim"] - 1 - 1, 0) + 1), Select(v["arg2_shape"], i) == Select(v["arg3_shape"], i)) for i in range(6)])), (Select(v["arg2_shape"], v["arg2_ndim"] - 1) <= v["arg1_ndim"]))))))
)

def rule_104_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 104
        rule_104(solver, {'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape, 'arg3_ndim': arg3_ndim, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_104(solver, {'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'], 'arg3_ndim': arg3['ndim'], 'arg3_shape': arg3['shape']}, neg)
