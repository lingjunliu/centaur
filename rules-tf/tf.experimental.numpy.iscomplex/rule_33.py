import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If shape is equal to 1 for all axes except for one axis, then it is complex (Rule 33)

rule_33 = lambda s, v, n=False: (
    s.add(Not(And(And((Or([And(i < (v["arg1_ndim"] - 1 + 1), True) for i in range(6)])), (And([Implies(j < (v["arg1_ndim"] - 1 + 1), If(i == j, True, Select(v["arg1_shape"], j) == 1)) for j in range(6)]))), (Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10)))) if n else
          And(And((Or([And(i < (v["arg1_ndim"] - 1 + 1), True) for i in range(6)])), (And([Implies(j < (v["arg1_ndim"] - 1 + 1), If(i == j, True, Select(v["arg1_shape"], j) == 1)) for j in range(6)]))), (Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10))))
)

def rule_33_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 33
        rule_33(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_33(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape']}, neg)
