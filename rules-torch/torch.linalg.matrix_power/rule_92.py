import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# A must be a tensor with float or complex dtype AND shape(A,0 (Rule 92)

rule_92 = lambda s, v, n=False: (
    s.add(Not(And((And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 11)), (If(And(And(v["arg1_ndim"] > 0, v["arg1_dtype"] < 9), v["arg1_ndim"] < 3), And(Select(v["arg1_shape"], 0) > 5, Select(v["arg1_shape"], 1) > 3), True)))) if n else
          And((And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 11)), (If(And(And(v["arg1_ndim"] > 0, v["arg1_dtype"] < 9), v["arg1_ndim"] < 3), And(Select(v["arg1_shape"], 0) > 5, Select(v["arg1_shape"], 1) > 3), True))))
)

def rule_92_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 92
        rule_92(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_92(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim']}, neg)
