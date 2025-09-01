import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If p is of complex type, then x must also have some element being complex numbers. (Rule 83)

rule_83 = lambda s, v, n=False: (
    s.add(Not(If((And(9 <= v["arg1_dtype"], v["arg1_dtype"] <= 10)), Or([And(i < (Select(v["arg2_shape"], 0) - 1 + 1), And((9 <= v["arg2_dtype"]), (v["arg2_dtype"] <= 10))) for i in range(6)]), True)) if n else
          If((And(9 <= v["arg1_dtype"], v["arg1_dtype"] <= 10)), Or([And(i < (Select(v["arg2_shape"], 0) - 1 + 1), And((9 <= v["arg2_dtype"]), (v["arg2_dtype"] <= 10))) for i in range(6)]), True))
)

def rule_83_func(arg1, arg2, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 83
        rule_83(solver, {'arg1_dtype': arg1_dtype, 'arg2_shape': arg2_shape, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_83(solver, {'arg1_dtype': arg1['dtype'], 'arg2_shape': arg2['shape'], 'arg2_dtype': arg2['dtype']}, neg)
