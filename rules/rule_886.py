import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if the tensor v_1 is 2 dimensional and the maximum shape between v_1 and v_2 is higher than 10 and the string "mean" is used ,the data type must be a float tensor (Rule 886)

rule_886 = lambda s, v, n=False: (
    s.add(Not(If(And(And((v["arg1_ndim"] == 2), ((Or(Select(v["arg1_shape"], 0) > 10, Select(v["arg1_shape"], 1) > 10)))), (v["arg2_value"] == 7)), (And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8)), False)) if n else
          If(And(And((v["arg1_ndim"] == 2), ((Or(Select(v["arg1_shape"], 0) > 10, Select(v["arg1_shape"], 1) > 10)))), (v["arg2_value"] == 7)), (And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8)), False))
)

def rule_886_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 886
        rule_886(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_886(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
