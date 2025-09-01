import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# in_channels, out_channels, groups, kernel_size, stride, padding, output_padding and dilation cannot be zero (Rule 80)

rule_80 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(v["arg1_value"] != 0, v["arg2_value"] != 0), v["arg3_value"] != 0), v["arg4_value"] != 0), v["arg5_value"] != 0), v["arg6_value"] != 0), v["arg7_value"] != 0), v["arg8_value"] != 0)) if n else
          And(And(And(And(And(And(And(v["arg1_value"] != 0, v["arg2_value"] != 0), v["arg3_value"] != 0), v["arg4_value"] != 0), v["arg5_value"] != 0), v["arg6_value"] != 0), v["arg7_value"] != 0), v["arg8_value"] != 0))
)

def rule_80_func(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))
    arg7 = next(iter(arg7.values()))
    arg8 = next(iter(arg8.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
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
        if not (isinstance(arg8, (int, np.integer)) and not isinstance(arg8, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')
        arg5_value = Int('arg5_value')
        arg6_value = Int('arg6_value')
        arg7_value = Int('arg7_value')
        arg8_value = Int('arg8_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == int(arg4))
        solver.add(arg5_value == int(arg5))
        solver.add(arg6_value == int(arg6))
        solver.add(arg7_value == int(arg7))
        solver.add(arg8_value == int(arg8))

        # Constraints for rule 80
        rule_80(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value, 'arg6_value': arg6_value, 'arg7_value': arg7_value, 'arg8_value': arg8_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_80(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value'], 'arg6_value': arg6['value'], 'arg7_value': arg7['value'], 'arg8_value': arg8['value']}, neg)
