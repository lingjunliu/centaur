import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Comprehensive Validation - 3D Image (Rule 24)

rule_24 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(And(v["arg1_ndim"] == 3, Select(v["arg1_shape"], 0) > 0), Select(v["arg1_shape"], 1) > 0), Select(v["arg1_shape"], 2) > 0), v["arg2_value"] >= 0), v["arg3_value"] > 0), Select(v["arg1_shape"], 1) >= v["arg3_value"] + v["arg4_value"]), v["arg5_value"] >= 0), v["arg6_value"] > 0), Select(v["arg1_shape"], 1) >= v["arg6_value"] + v["arg7_value"])) if n else
          And(And(And(And(And(And(And(And(And(v["arg1_ndim"] == 3, Select(v["arg1_shape"], 0) > 0), Select(v["arg1_shape"], 1) > 0), Select(v["arg1_shape"], 2) > 0), v["arg2_value"] >= 0), v["arg3_value"] > 0), Select(v["arg1_shape"], 1) >= v["arg3_value"] + v["arg4_value"]), v["arg5_value"] >= 0), v["arg6_value"] > 0), Select(v["arg1_shape"], 1) >= v["arg6_value"] + v["arg7_value"]))
)

def rule_24_func(arg1, arg2, arg3, arg4, arg5, arg6, arg7, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))
    arg7 = next(iter(arg7.values()))

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
        if not (isinstance(arg5, (int, np.integer)) and not isinstance(arg5, bool)):
            return False
        if not (isinstance(arg6, (int, np.integer)) and not isinstance(arg6, bool)):
            return False
        if not (isinstance(arg7, (int, np.integer)) and not isinstance(arg7, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')
        arg5_value = Int('arg5_value')
        arg6_value = Int('arg6_value')
        arg7_value = Int('arg7_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == int(arg4))
        solver.add(arg5_value == int(arg5))
        solver.add(arg6_value == int(arg6))
        solver.add(arg7_value == int(arg7))

        # Constraints for rule 24
        rule_24(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value, 'arg6_value': arg6_value, 'arg7_value': arg7_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_24(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value'], 'arg6_value': arg6['value'], 'arg7_value': arg7['value']}, neg)
