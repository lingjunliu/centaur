import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# data type of values should be of appropriate type to align with indices. Also, if indices is non-empty then they must be compatible datatypes (Rule 46)

rule_46 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_shape"], 0) > 0, (If(And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5), And(1 <= v["arg2_dtype"], v["arg2_dtype"] <= 5), If(And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8), And(6 <= v["arg2_dtype"], v["arg2_dtype"] <= 8), True))), True)) if n else
          If(Select(v["arg1_shape"], 0) > 0, (If(And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5), And(1 <= v["arg2_dtype"], v["arg2_dtype"] <= 5), If(And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8), And(6 <= v["arg2_dtype"], v["arg2_dtype"] <= 8), True))), True))
)

def rule_46_func(arg1, arg2, solver=None, neg=False):
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
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 46
        rule_46(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_46(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg2_dtype': arg2['dtype']}, neg)
