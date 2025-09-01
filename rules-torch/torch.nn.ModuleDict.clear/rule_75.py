import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If dtype of v_1 is not none and shape of dimension 0 is not zero and min is greater than zero and maximum needs to be smaller than certain number, lets say 20. (Rule 75)

rule_75 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_dtype"] != 6, Select(v["arg1_shape"], 0) != 0), Select(v["arg1_range"], 0) > 0), Select(v["arg1_range"], 1) < 20, True)) if n else
          If(And(And(v["arg1_dtype"] != 6, Select(v["arg1_shape"], 0) != 0), Select(v["arg1_range"], 0) > 0), Select(v["arg1_range"], 1) < 20, True))
)

def rule_75_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 75
        rule_75(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_75(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range'], 'arg1_shape': arg1['shape']}, neg)
