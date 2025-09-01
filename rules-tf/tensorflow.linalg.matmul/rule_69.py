import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If sparse and a is complex, shape must be valid and a dimension has size 1 (Rule 69)

rule_69 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_value"], (Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10))), And((And(Select(v["arg1_shape"], 0) > 0, Select(v["arg1_shape"], 1) > 0)), (Or(Select(v["arg1_shape"], 0) == 1, Select(v["arg1_shape"], 1) == 1))), True)) if n else
          If(And(v["arg2_value"], (Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10))), And((And(Select(v["arg1_shape"], 0) > 0, Select(v["arg1_shape"], 1) > 0)), (Or(Select(v["arg1_shape"], 0) == 1, Select(v["arg1_shape"], 1) == 1))), True))
)

def rule_69_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Bool('arg2_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)

        # Constraints for rule 69
        rule_69(solver, {'arg1_shape': arg1_shape, 'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_69(solver, {'arg1_shape': arg1['shape'], 'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
