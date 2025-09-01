import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If a size is supplied then the number of values has to be less than the product of the size and the first value should be less than 100 and the min should be divisble by 3 and all data should be int (Rule 121)

rule_121 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_length"] > 0, And(And((And(Select(v["arg1_shape"], 0) <= (Or(Or(Select(v["arg2_values"], 0) * Select(v["arg2_values"], 1), Select(v["arg2_values"], 0) > 1000000), Select(v["arg2_values"], 1) > 1000000)), Select(v["arg2_values"], 0) < 100)), (Select(v["arg1_range"], 0) % 3 == 0)), (Or(v["arg1_dtype"] == 3, v["arg1_dtype"] == 4))), True)) if n else
          If(v["arg2_length"] > 0, And(And((And(Select(v["arg1_shape"], 0) <= (Or(Or(Select(v["arg2_values"], 0) * Select(v["arg2_values"], 1), Select(v["arg2_values"], 0) > 1000000), Select(v["arg2_values"], 1) > 1000000)), Select(v["arg2_values"], 0) < 100)), (Select(v["arg1_range"], 0) % 3 == 0)), (Or(v["arg1_dtype"] == 3, v["arg1_dtype"] == 4))), True))
)

def rule_121_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 121
        rule_121(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg2_length': arg2_length, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_121(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg2_length': arg2['length'], 'arg2_values': arg2['values']}, neg)
