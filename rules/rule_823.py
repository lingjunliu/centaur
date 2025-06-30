import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If the shape of first dimension is bigger than 5, and string is sum or mean or max, then dtype has to be float or complex (Rule 823)

rule_823 = lambda s, v, n=False: (
    s.add(Not(If(And((Select(v["arg1_shape"], 0) > 5), (Or(Or(v["arg2_value"] == 8, v["arg2_value"] == 7), v["arg2_value"] == 9))), (And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 10)), False)) if n else
          If(And((Select(v["arg1_shape"], 0) > 5), (Or(Or(v["arg2_value"] == 8, v["arg2_value"] == 7), v["arg2_value"] == 9))), (And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 10)), False))
)

def rule_823_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_value = String('arg2_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == list_of_string_values.index(arg2))

        # Constraints for rule 823
        rule_823(solver, {'arg1_shape': arg1_shape, 'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_823(solver, {'arg1_shape': arg1['shape'], 'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
