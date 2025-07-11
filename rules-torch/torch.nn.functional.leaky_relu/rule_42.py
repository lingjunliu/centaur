import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# input must be a tensor, and its dtype should not be Char, and if inplace is true, input must be a floating point tensor, negative_slope should be between 0 and 1, negative slope is greater than the minimum element of the input, and the shape should be valid, and also check the rank of the input tensor is less than 5, and if input tensor is integer type, inplace must be false (Rule 42)

rule_42 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(v["arg1_dtype"] != 0, (If(v["arg3_value"], (Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8)), False))), 0 <= v["arg2_value"]), v["arg2_value"] <= 1), v["arg2_value"] > Select(v["arg1_range"], 0)), And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(And(Select(v["arg1_shape"], i) < 100, v["arg1_ndim"] < 5), (If(v["arg1_dtype"] < 6, v["arg3_value"] == False, False)))) for i in range(6)]))) if n else
          And(And(And(And(And(v["arg1_dtype"] != 0, (If(v["arg3_value"], (Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8)), False))), 0 <= v["arg2_value"]), v["arg2_value"] <= 1), v["arg2_value"] > Select(v["arg1_range"], 0)), And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(And(Select(v["arg1_shape"], i) < 100, v["arg1_ndim"] < 5), (If(v["arg1_dtype"] < 6, v["arg3_value"] == False, False)))) for i in range(6)])))
)

def rule_42_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Real('arg2_value')
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)

        # Constraints for rule 42
        rule_42(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg1_range': arg1_range, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_42(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg1_range': arg1['range'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
