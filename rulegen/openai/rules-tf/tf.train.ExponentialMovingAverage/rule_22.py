import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# full parameter check for float-decay signature (Rule 22)

rule_22 = lambda s, v, n=False: (
    s.add(Not(And(And(And((And(0 < v["arg1_value"], v["arg1_value"] < 1)), (And(And(v["arg2_ndim"] == 0, 1 <= v["arg2_dtype"]), v["arg2_dtype"] <= 5))), (Or(v["arg3_value"] == True, v["arg3_value"] == False))), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg4_value"] == 0, v["arg4_value"] == 1), v["arg4_value"] == 2), v["arg4_value"] == 3), v["arg4_value"] == 4), v["arg4_value"] == 5), v["arg4_value"] == 6), v["arg4_value"] == 7), v["arg4_value"] == 8), v["arg4_value"] == 9), v["arg4_value"] == 10), v["arg4_value"] == 11), v["arg4_value"] == 12), v["arg4_value"] == 13), v["arg4_value"] == 14), v["arg4_value"] == 15), v["arg4_value"] == 16), v["arg4_value"] == 17), v["arg4_value"] == 18), v["arg4_value"] == 19), v["arg4_value"] == 20), v["arg4_value"] == 21), v["arg4_value"] == 22), v["arg4_value"] == 23), v["arg4_value"] == 24), v["arg4_value"] == 25)))) if n else
          And(And(And((And(0 < v["arg1_value"], v["arg1_value"] < 1)), (And(And(v["arg2_ndim"] == 0, 1 <= v["arg2_dtype"]), v["arg2_dtype"] <= 5))), (Or(v["arg3_value"] == True, v["arg3_value"] == False))), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg4_value"] == 0, v["arg4_value"] == 1), v["arg4_value"] == 2), v["arg4_value"] == 3), v["arg4_value"] == 4), v["arg4_value"] == 5), v["arg4_value"] == 6), v["arg4_value"] == 7), v["arg4_value"] == 8), v["arg4_value"] == 9), v["arg4_value"] == 10), v["arg4_value"] == 11), v["arg4_value"] == 12), v["arg4_value"] == 13), v["arg4_value"] == 14), v["arg4_value"] == 15), v["arg4_value"] == 16), v["arg4_value"] == 17), v["arg4_value"] == 18), v["arg4_value"] == 19), v["arg4_value"] == 20), v["arg4_value"] == 21), v["arg4_value"] == 22), v["arg4_value"] == 23), v["arg4_value"] == 24), v["arg4_value"] == 25))))
)

def rule_22_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, bool):
            return False
        if not isinstance(arg4, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')
        arg3_value = Bool('arg3_value')
        arg4_value = String('arg4_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == list_of_string_values_tf.index(arg4))

        # Constraints for rule 22
        rule_22(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_22(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
