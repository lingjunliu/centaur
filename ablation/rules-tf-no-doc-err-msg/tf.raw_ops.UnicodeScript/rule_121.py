import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If a float value is less than 0 and the shape is less than 5, then at least one shape of the tensor must be greater than 3 and smaller than 10 and it's integer type (Rule 121)

rule_121 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] < 0, Select(v["arg2_shape"], 0) < 5), Or([And(i < (v["arg2_ndim"] - 1 + 1), And((And(Select(v["arg2_shape"], i) > 3, Select(v["arg2_shape"], i) < 10)), (And(1 <= v["arg2_dtype"], v["arg2_dtype"] <= 5)))) for i in range(6)]), True)) if n else
          If(And(v["arg1_value"] < 0, Select(v["arg2_shape"], 0) < 5), Or([And(i < (v["arg2_ndim"] - 1 + 1), And((And(Select(v["arg2_shape"], i) > 3, Select(v["arg2_shape"], i) < 10)), (And(1 <= v["arg2_dtype"], v["arg2_dtype"] <= 5)))) for i in range(6)]), True))
)

def rule_121_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 121
        rule_121(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_121(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'], 'arg2_dtype': arg2['dtype']}, neg)
