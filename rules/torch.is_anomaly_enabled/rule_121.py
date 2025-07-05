import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If shape of dimension is less than 10 and greater than 5, then its dtype must be integer number or string or bool, or dtype is complex, or dtype is float16 or shape should not be 7 (Rule 121)

rule_121 = lambda s, v, n=False: (
    s.add(Not(If(And(Select(v["arg1_shape"], v["arg2_value"]) < 10, Select(v["arg1_shape"], v["arg2_value"]) > 5), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 5), v["arg1_dtype"] == 11), v["arg1_dtype"] == 0), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10), v["arg1_dtype"] == 6), Select(v["arg1_shape"], v["arg2_value"]) != 7)), False)) if n else
          If(And(Select(v["arg1_shape"], v["arg2_value"]) < 10, Select(v["arg1_shape"], v["arg2_value"]) > 5), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 5), v["arg1_dtype"] == 11), v["arg1_dtype"] == 0), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10), v["arg1_dtype"] == 6), Select(v["arg1_shape"], v["arg2_value"]) != 7)), False))
)

def rule_121_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 121
        rule_121(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_121(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)
