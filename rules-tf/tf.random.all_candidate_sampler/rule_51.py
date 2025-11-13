import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Combined type, size, and value checks for API parameters (Rule 51)

rule_51 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And((v["arg1_dtype"] == 4), (Or((And(v["arg1_ndim"] == 1, v["arg2_value"] == 1)), (And(v["arg1_ndim"] == 2, Select(v["arg1_shape"], 1) == v["arg2_value"]))))), (And(1 <= v["arg3_value"], v["arg3_value"] <= 2147483647))), (And(1 <= v["arg2_value"], v["arg2_value"] <= 2147483647))), (v["arg2_value"] <= v["arg3_value"])), (And(-2147483648 <= v["arg4_value"], v["arg4_value"] <= 2147483647)))) if n else
          And(And(And(And(And((v["arg1_dtype"] == 4), (Or((And(v["arg1_ndim"] == 1, v["arg2_value"] == 1)), (And(v["arg1_ndim"] == 2, Select(v["arg1_shape"], 1) == v["arg2_value"]))))), (And(1 <= v["arg3_value"], v["arg3_value"] <= 2147483647))), (And(1 <= v["arg2_value"], v["arg2_value"] <= 2147483647))), (v["arg2_value"] <= v["arg3_value"])), (And(-2147483648 <= v["arg4_value"], v["arg4_value"] <= 2147483647))))
)

def rule_51_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == int(arg4))

        # Constraints for rule 51
        rule_51(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_51(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
