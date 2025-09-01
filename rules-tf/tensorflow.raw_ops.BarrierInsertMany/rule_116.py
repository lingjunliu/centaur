import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If values tensor is of type bool, then its shape should be equal to the component index and its value must be true and it can not be the default dtype and component_index must not be a very large number and min of keys and max of keys should be greater or less than 10 and the sum of dimesnsions of values should not exceed 1000 and component_index should be less than 10 (Rule 116)

rule_116 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 0, And(And(And(And(And(Select(v["arg1_shape"], 0) == v["arg2_value"], Select(v["arg1_range"], 0) == True), v["arg1_dtype"] != 12), v["arg2_value"] < 500), (Or(Select(v["arg3_range"], 0) > 10, Select(v["arg3_range"], 1) < 10))), And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(Select(v["arg1_shape"], i) < 1000, v["arg2_value"] < 10)) for i in range(6)])), True)) if n else
          If(v["arg1_dtype"] == 0, And(And(And(And(And(Select(v["arg1_shape"], 0) == v["arg2_value"], Select(v["arg1_range"], 0) == True), v["arg1_dtype"] != 12), v["arg2_value"] < 500), (Or(Select(v["arg3_range"], 0) > 10, Select(v["arg3_range"], 1) < 10))), And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(Select(v["arg1_shape"], i) < 1000, v["arg2_value"] < 10)) for i in range(6)])), True))
)

def rule_116_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_range = Array('arg3_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == int(arg2))
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 116
        rule_116(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg1_range': arg1_range, 'arg2_value': arg2_value, 'arg3_range': arg3_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_116(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg1_range': arg1['range'], 'arg2_value': arg2['value'], 'arg3_range': arg3['range']}, neg)
