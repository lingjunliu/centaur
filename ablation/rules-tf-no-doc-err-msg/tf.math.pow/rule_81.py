import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If tensor's ndim is less than 2 then min and max values should be equal or their difference should be less than one and dtype is not boolean and then the min value should be smaller than 100 and max greater than 50 and then shape(0 (Rule 81)

rule_81 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] < 2, And(And(And(And((Or(Select(v["arg1_range"], 0) == Select(v["arg1_range"], 1), Select(v["arg1_range"], 1) - Select(v["arg1_range"], 0) < 1)), v["arg1_dtype"] != 0), Select(v["arg1_range"], 0) < 100), Select(v["arg1_range"], 1) > 50), Select(v["arg1_shape"], 0) == 0), True)) if n else
          If(v["arg1_ndim"] < 2, And(And(And(And((Or(Select(v["arg1_range"], 0) == Select(v["arg1_range"], 1), Select(v["arg1_range"], 1) - Select(v["arg1_range"], 0) < 1)), v["arg1_dtype"] != 0), Select(v["arg1_range"], 0) < 100), Select(v["arg1_range"], 1) > 50), Select(v["arg1_shape"], 0) == 0), True))
)

def rule_81_func(arg1, solver=None, neg=False):
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
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 81
        rule_81(solver, {'arg1_ndim': arg1_ndim, 'arg1_range': arg1_range, 'arg1_shape': arg1_shape, 'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_81(solver, {'arg1_ndim': arg1['ndim'], 'arg1_range': arg1['range'], 'arg1_shape': arg1['shape'], 'arg1_dtype': arg1['dtype']}, neg)
