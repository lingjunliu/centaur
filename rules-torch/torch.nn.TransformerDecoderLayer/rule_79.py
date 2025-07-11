import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# Complete condition check for all parameters with a > comparison (Rule 79)

rule_79 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And((And(And(And(v["arg1_value"] > 0, v["arg2_value"] > 0), v["arg1_value"] % v["arg2_value"] == 0), v["arg3_value"] > v["arg1_value"])), (And(v["arg4_value"] >= 0, v["arg4_value"] <= 1))), (Or(v["arg5_value"] == 12, v["arg5_value"] == 18))), (v["arg6_value"] > 0)), (Or(v["arg7_value"] == True, v["arg7_value"] == False))), (Or(v["arg8_value"] == True, v["arg8_value"] == False))), (Or(v["arg9_value"] == True, v["arg9_value"] == False)))) if n else
          And(And(And(And(And(And((And(And(And(v["arg1_value"] > 0, v["arg2_value"] > 0), v["arg1_value"] % v["arg2_value"] == 0), v["arg3_value"] > v["arg1_value"])), (And(v["arg4_value"] >= 0, v["arg4_value"] <= 1))), (Or(v["arg5_value"] == 12, v["arg5_value"] == 18))), (v["arg6_value"] > 0)), (Or(v["arg7_value"] == True, v["arg7_value"] == False))), (Or(v["arg8_value"] == True, v["arg8_value"] == False))), (Or(v["arg9_value"] == True, v["arg9_value"] == False))))
)

def rule_79_func(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))
    arg7 = next(iter(arg7.values()))
    arg8 = next(iter(arg8.values()))
    arg9 = next(iter(arg9.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not isinstance(arg4, (float, np.floating)):
            return False
        if not isinstance(arg5, str):
            return False
        if not isinstance(arg6, (float, np.floating)):
            return False
        if not isinstance(arg7, bool):
            return False
        if not isinstance(arg8, bool):
            return False
        if not isinstance(arg9, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')
        arg4_value = Real('arg4_value')
        arg5_value = String('arg5_value')
        arg6_value = Real('arg6_value')
        arg7_value = Bool('arg7_value')
        arg8_value = Bool('arg8_value')
        arg9_value = Bool('arg9_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == arg4)
        solver.add(arg5_value == list_of_string_values_torch.torch.index(arg5))
        solver.add(arg6_value == arg6)
        solver.add(arg7_value == arg7)
        solver.add(arg8_value == arg8)
        solver.add(arg9_value == arg9)

        # Constraints for rule 79
        rule_79(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value, 'arg6_value': arg6_value, 'arg7_value': arg7_value, 'arg8_value': arg8_value, 'arg9_value': arg9_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_79(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value'], 'arg6_value': arg6['value'], 'arg7_value': arg7['value'], 'arg8_value': arg8['value'], 'arg9_value': arg9['value']}, neg)
