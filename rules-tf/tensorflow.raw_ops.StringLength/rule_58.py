import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if input is a string tensor with ndim=N, then the output is an int32 tensor with ndim=N and same shape (Rule 58)

rule_58 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_dtype"] == 11, And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) == Select(v["arg1_shape"], i)) for i in range(6)]))) if n else
          And(v["arg1_dtype"] == 11, And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) == Select(v["arg1_shape"], i)) for i in range(6)])))
)

def rule_58_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 58
        rule_58(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_58(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim']}, neg)
