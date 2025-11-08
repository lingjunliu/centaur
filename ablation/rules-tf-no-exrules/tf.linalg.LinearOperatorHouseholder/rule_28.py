import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Check boolean flags, reflection_axis properties and name (Rule 28)

rule_28 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(v["arg2_value"] == True, v["arg4_value"] == True), v["arg3_value"] == False), v["arg1_ndim"] >= 1), (Or(v["arg1_dtype"] == 8, v["arg1_dtype"] == 9))), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg5_value"] == 6, v["arg5_value"] == 7), v["arg5_value"] == 8), v["arg5_value"] == 9), v["arg5_value"] == 10), v["arg5_value"] == 11), v["arg5_value"] == 12), v["arg5_value"] == 13), v["arg5_value"] == 14), v["arg5_value"] == 15), v["arg5_value"] == 16), v["arg5_value"] == 17), v["arg5_value"] == 18), v["arg5_value"] == 19), v["arg5_value"] == 20), v["arg5_value"] == 21), v["arg5_value"] == 22), v["arg5_value"] == 23), v["arg5_value"] == 24), v["arg5_value"] == 25)))) if n else
          And(And(And(And(And(v["arg2_value"] == True, v["arg4_value"] == True), v["arg3_value"] == False), v["arg1_ndim"] >= 1), (Or(v["arg1_dtype"] == 8, v["arg1_dtype"] == 9))), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg5_value"] == 6, v["arg5_value"] == 7), v["arg5_value"] == 8), v["arg5_value"] == 9), v["arg5_value"] == 10), v["arg5_value"] == 11), v["arg5_value"] == 12), v["arg5_value"] == 13), v["arg5_value"] == 14), v["arg5_value"] == 15), v["arg5_value"] == 16), v["arg5_value"] == 17), v["arg5_value"] == 18), v["arg5_value"] == 19), v["arg5_value"] == 20), v["arg5_value"] == 21), v["arg5_value"] == 22), v["arg5_value"] == 23), v["arg5_value"] == 24), v["arg5_value"] == 25))))
)

def rule_28_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, bool):
            return False
        if not isinstance(arg4, bool):
            return False
        if not isinstance(arg5, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Bool('arg2_value')
        arg3_value = Bool('arg3_value')
        arg4_value = Bool('arg4_value')
        arg5_value = String('arg5_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == arg4)
        solver.add(arg5_value == list_of_string_values_tf.index(arg5))

        # Constraints for rule 28
        rule_28(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_28(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value']}, neg)
