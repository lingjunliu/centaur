import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If at least one tensor has ndim > 1 and dtypes are integer then product of shapes should not overflow (Rule 51)

rule_51 = lambda s, v, n=False: (
    s.add(Not(If(And(And((Or(v["arg1_ndim"] > 1, v["arg2_ndim"] > 1)), (And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5))), (And(1 <= v["arg2_dtype"], v["arg2_dtype"] <= 5))), (And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i)) for i in range(6)])) * (And([Implies(j < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], j)) for j in range(6)])) < 2147483647, True)) if n else
          If(And(And((Or(v["arg1_ndim"] > 1, v["arg2_ndim"] > 1)), (And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5))), (And(1 <= v["arg2_dtype"], v["arg2_dtype"] <= 5))), (And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i)) for i in range(6)])) * (And([Implies(j < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], j)) for j in range(6)])) < 2147483647, True))
)

def rule_51_func(arg1, arg2, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 51
        rule_51(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_51(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_shape': arg2['shape']}, neg)
