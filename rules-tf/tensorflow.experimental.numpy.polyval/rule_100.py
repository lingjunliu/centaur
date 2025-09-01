import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if there is at least one integer element in x, then p cannot be of complex type  (Rule 100)

rule_100 = lambda s, v, n=False: (
    s.add(Not(If(Or([And(i < (Select(v["arg2_shape"], 0) - 1 + 1), (And(1 <= v["arg2_dtype"], v["arg2_dtype"] <= 5))) for i in range(6)]), (Or(9 > v["arg1_dtype"], v["arg1_dtype"] > 10)), True)) if n else
          If(Or([And(i < (Select(v["arg2_shape"], 0) - 1 + 1), (And(1 <= v["arg2_dtype"], v["arg2_dtype"] <= 5))) for i in range(6)]), (Or(9 > v["arg1_dtype"], v["arg1_dtype"] > 10)), True))
)

def rule_100_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 100
        rule_100(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_100(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg2_shape': arg2['shape']}, neg)
