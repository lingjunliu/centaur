import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# valid 2D block count L consistent with output_size/stride/padding/dilation/kernel_size (Rule 12)

rule_12 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(And(v["arg1_length"] == 2, v["arg2_length"] == 2), v["arg3_length"] == 2), v["arg4_length"] == 2), v["arg5_length"] == 2), (Select(v["arg1_values"], 0) + 2 * Select(v["arg4_values"], 0) - Select(v["arg3_values"], 0) * (Select(v["arg2_values"], 0) - 1) - 1) >= 0), (Select(v["arg1_values"], 1) + 2 * Select(v["arg4_values"], 1) - Select(v["arg3_values"], 1) * (Select(v["arg2_values"], 1) - 1) - 1) >= 0), ((Select(v["arg1_values"], 0) + 2 * Select(v["arg4_values"], 0) - Select(v["arg3_values"], 0) * (Select(v["arg2_values"], 0) - 1) - 1) % Select(v["arg5_values"], 0)) == 0), ((Select(v["arg1_values"], 1) + 2 * Select(v["arg4_values"], 1) - Select(v["arg3_values"], 1) * (Select(v["arg2_values"], 1) - 1) - 1) % Select(v["arg5_values"], 1)) == 0), Select(v["arg6_shape"], 2) == ((Select(v["arg1_values"], 0) + 2 * Select(v["arg4_values"], 0) - Select(v["arg3_values"], 0) * (Select(v["arg2_values"], 0) - 1) - 1) / Select(v["arg5_values"], 0) + 1) * ((Select(v["arg1_values"], 1) + 2 * Select(v["arg4_values"], 1) - Select(v["arg3_values"], 1) * (Select(v["arg2_values"], 1) - 1) - 1) / Select(v["arg5_values"], 1) + 1))) if n else
          And(And(And(And(And(And(And(And(And(v["arg1_length"] == 2, v["arg2_length"] == 2), v["arg3_length"] == 2), v["arg4_length"] == 2), v["arg5_length"] == 2), (Select(v["arg1_values"], 0) + 2 * Select(v["arg4_values"], 0) - Select(v["arg3_values"], 0) * (Select(v["arg2_values"], 0) - 1) - 1) >= 0), (Select(v["arg1_values"], 1) + 2 * Select(v["arg4_values"], 1) - Select(v["arg3_values"], 1) * (Select(v["arg2_values"], 1) - 1) - 1) >= 0), ((Select(v["arg1_values"], 0) + 2 * Select(v["arg4_values"], 0) - Select(v["arg3_values"], 0) * (Select(v["arg2_values"], 0) - 1) - 1) % Select(v["arg5_values"], 0)) == 0), ((Select(v["arg1_values"], 1) + 2 * Select(v["arg4_values"], 1) - Select(v["arg3_values"], 1) * (Select(v["arg2_values"], 1) - 1) - 1) % Select(v["arg5_values"], 1)) == 0), Select(v["arg6_shape"], 2) == ((Select(v["arg1_values"], 0) + 2 * Select(v["arg4_values"], 0) - Select(v["arg3_values"], 0) * (Select(v["arg2_values"], 0) - 1) - 1) / Select(v["arg5_values"], 0) + 1) * ((Select(v["arg1_values"], 1) + 2 * Select(v["arg4_values"], 1) - Select(v["arg3_values"], 1) * (Select(v["arg2_values"], 1) - 1) - 1) / Select(v["arg5_values"], 1) + 1)))
)

def rule_12_func(arg1, arg2, arg3, arg4, arg5, arg6, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False
        if not (isinstance(arg4, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False
        if not (isinstance(arg5, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg5)):
            return False
        if not isinstance(arg6, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())
        arg4_length = Int('arg4_length')
        arg4_values = Array('arg4_values', IntSort(), IntSort())
        arg5_length = Int('arg5_length')
        arg5_values = Array('arg5_values', IntSort(), IntSort())
        arg6_shape = Array('arg6_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])
        solver.add(arg4_length == len(arg4))
        for i in range(len(arg4)):
            arg4_values = Store(arg4_values, i, arg4[i])
        solver.add(arg5_length == len(arg5))
        for i in range(len(arg5)):
            arg5_values = Store(arg5_values, i, arg5[i])
        for i in range(arg6.ndim):
            arg6_shape = Store(arg6_shape, i, arg6.shape[i])

        # Constraints for rule 12
        rule_12(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length, 'arg2_values': arg2_values, 'arg2_length': arg2_length, 'arg3_values': arg3_values, 'arg3_length': arg3_length, 'arg4_values': arg4_values, 'arg4_length': arg4_length, 'arg5_values': arg5_values, 'arg5_length': arg5_length, 'arg6_shape': arg6_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_12(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length'], 'arg2_values': arg2['values'], 'arg2_length': arg2['length'], 'arg3_values': arg3['values'], 'arg3_length': arg3['length'], 'arg4_values': arg4['values'], 'arg4_length': arg4['length'], 'arg5_values': arg5['values'], 'arg5_length': arg5['length'], 'arg6_shape': arg6['shape']}, neg)
