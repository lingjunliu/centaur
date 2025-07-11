import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# check for float range and multiply by an int and compare to number, check for dtype and if the dimensions are equal (Rule 68)

rule_68 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(v["arg1_value"] * v["arg2_value"] >= 0.0, v["arg1_value"] * v["arg2_value"] <= 10.0), v["arg3_value"] > 5), v["arg4_value"] == 7), v["arg5_ndim"] == v["arg2_value"])) if n else
          And(And(And(And(v["arg1_value"] * v["arg2_value"] >= 0.0, v["arg1_value"] * v["arg2_value"] <= 10.0), v["arg3_value"] > 5), v["arg4_value"] == 7), v["arg5_ndim"] == v["arg2_value"]))
)

def rule_68_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg4, torch.dtype) or isinstance(arg4, tf.dtypes.DType)):
            return False
        if not isinstance(arg5, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')
        arg5_ndim = Int('arg5_ndim')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == list_of_available_dtypes.index(np_dtype(arg4)))
        solver.add(arg5_ndim == arg5.ndim)

        # Constraints for rule 68
        rule_68(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_ndim': arg5_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_68(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_ndim': arg5['ndim']}, neg)
