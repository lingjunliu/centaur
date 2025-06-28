import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if string is 'none' and tensor is of float or integer and shape of the tensor on any axis is different to 1, then it is always false (Rule 324)

rule_324 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg2_value"] == 6, (Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 5), v["arg1_dtype"] == 6), v["arg1_dtype"] == 7), v["arg1_dtype"] == 8))), (Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) != 1) for i in range(6)]))), False, False)) if n else
          If(And(And(v["arg2_value"] == 6, (Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 5), v["arg1_dtype"] == 6), v["arg1_dtype"] == 7), v["arg1_dtype"] == 8))), (Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) != 1) for i in range(6)]))), False, False))
)

def rule_324_func(arg1, arg2, solver=None, neg=False):
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
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == list_of_string_values.index(arg2))

        # Constraints for rule 324
        rule_324(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_324(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)
