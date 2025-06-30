import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if ndim > 0 and max < 0 and string is constant, tensor should have float data type, also check condition that max * shape must be greater than or equals min for any i (Rule 272)

rule_272 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_ndim"] > 0, Select(v["arg1_range"], 1) < 0), v["arg2_value"] == 10), And((Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8)), Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_range"], 1) * Select(v["arg1_shape"], i) >= Select(v["arg1_range"], 0)) for i in range(6)])), False)) if n else
          If(And(And(v["arg1_ndim"] > 0, Select(v["arg1_range"], 1) < 0), v["arg2_value"] == 10), And((Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8)), Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_range"], 1) * Select(v["arg1_shape"], i) >= Select(v["arg1_range"], 0)) for i in range(6)])), False))
)

def rule_272_func(arg1, arg2, solver=None, neg=False):
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
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == list_of_string_values.index(arg2))

        # Constraints for rule 272
        rule_272(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg1_range': arg1_range, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_272(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg1_range': arg1['range'], 'arg2_value': arg2['value']}, neg)
