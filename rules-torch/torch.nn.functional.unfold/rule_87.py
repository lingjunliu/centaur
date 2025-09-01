import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Kernel can not be zero sized, can not be enormous compared to the tensor, the combination with dilation needs to be checked, and image dimension size can not be small. Padding value must be reasonable. (Rule 87)

rule_87 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(Select(v["arg1_values"], 0) > 0, Select(v["arg1_values"], 1) > 0), Select(v["arg2_shape"], 2) >= Select(v["arg1_values"], 0) + (Select(v["arg1_values"], 0) - 1) * (v["arg3_value"] - 1)), Select(v["arg2_shape"], 3) >= Select(v["arg1_values"], 1) + (Select(v["arg1_values"], 1) - 1) * (v["arg3_value"] - 1)), Select(v["arg1_values"], 0) < 1000), Select(v["arg1_values"], 1) < 1000), Select(v["arg2_shape"], 2) > 0), Select(v["arg2_shape"], 3) > 0), v["arg4_value"] < 1000)) if n else
          And(And(And(And(And(And(And(And(Select(v["arg1_values"], 0) > 0, Select(v["arg1_values"], 1) > 0), Select(v["arg2_shape"], 2) >= Select(v["arg1_values"], 0) + (Select(v["arg1_values"], 0) - 1) * (v["arg3_value"] - 1)), Select(v["arg2_shape"], 3) >= Select(v["arg1_values"], 1) + (Select(v["arg1_values"], 1) - 1) * (v["arg3_value"] - 1)), Select(v["arg1_values"], 0) < 1000), Select(v["arg1_values"], 1) < 1000), Select(v["arg2_shape"], 2) > 0), Select(v["arg2_shape"], 3) > 0), v["arg4_value"] < 1000))
)

def rule_87_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == int(arg4))

        # Constraints for rule 87
        rule_87(solver, {'arg1_values': arg1_values, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_87(solver, {'arg1_values': arg1['values'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
