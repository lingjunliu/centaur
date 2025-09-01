import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Combined: Shape at split dimension must be divisible by 2. Dimension has to be valid. Split dimension should be provided as an Integer. Dimension > 0. Tensor has appropriate dtype (Rule 49)

rule_49 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And((0 - v["arg2_ndim"]) <= v["arg1_value"], v["arg1_value"] < v["arg2_ndim"]), Select(v["arg2_shape"], (If(v["arg1_value"] < 0, v["arg2_ndim"] + v["arg1_value"], v["arg1_value"]))) % 2 == 0), v["arg2_ndim"] > 0), Select(v["arg2_shape"], (If(v["arg1_value"] < 0, v["arg2_ndim"] + v["arg1_value"], v["arg1_value"]))) >= 2), (Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg2_dtype"] == 1, v["arg2_dtype"] == 2), v["arg2_dtype"] == 3), v["arg2_dtype"] == 4), v["arg2_dtype"] == 5), v["arg2_dtype"] == 6), v["arg2_dtype"] == 7), v["arg2_dtype"] == 8), v["arg2_dtype"] == 9), v["arg2_dtype"] == 10)))) if n else
          And(And(And(And(And((0 - v["arg2_ndim"]) <= v["arg1_value"], v["arg1_value"] < v["arg2_ndim"]), Select(v["arg2_shape"], (If(v["arg1_value"] < 0, v["arg2_ndim"] + v["arg1_value"], v["arg1_value"]))) % 2 == 0), v["arg2_ndim"] > 0), Select(v["arg2_shape"], (If(v["arg1_value"] < 0, v["arg2_ndim"] + v["arg1_value"], v["arg1_value"]))) >= 2), (Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg2_dtype"] == 1, v["arg2_dtype"] == 2), v["arg2_dtype"] == 3), v["arg2_dtype"] == 4), v["arg2_dtype"] == 5), v["arg2_dtype"] == 6), v["arg2_dtype"] == 7), v["arg2_dtype"] == 8), v["arg2_dtype"] == 9), v["arg2_dtype"] == 10))))
)

def rule_49_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 49
        rule_49(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_49(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim']}, neg)
