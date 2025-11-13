import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# At least one of all the boolean options should be true or false, encompassing all boolean types. (Rule 67)

rule_67 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or((Or(v["arg1_value"] == True, v["arg1_value"] == False)), (Or(v["arg2_value"] == True, v["arg2_value"] == False))), (Or(v["arg3_value"] == True, v["arg3_value"] == False))), (Or(v["arg4_value"] == True, v["arg4_value"] == False))), (Or(v["arg5_value"] == True, v["arg5_value"] == False))), (Or(v["arg6_value"] == True, v["arg6_value"] == False))), (Or(v["arg7_value"] == True, v["arg7_value"] == False))), (Or(v["arg8_value"] == True, v["arg8_value"] == False))), (Or(v["arg9_value"] == True, v["arg9_value"] == False))), (Or(v["arg10_value"] == True, v["arg10_value"] == False))), (Or(v["arg11_value"] == True, v["arg11_value"] == False))), (Or(v["arg12_value"] == True, v["arg12_value"] == False))), (Or(v["arg13_value"] == True, v["arg13_value"] == False))), (Or(v["arg14_value"] == True, v["arg14_value"] == False)))) if n else
          Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or((Or(v["arg1_value"] == True, v["arg1_value"] == False)), (Or(v["arg2_value"] == True, v["arg2_value"] == False))), (Or(v["arg3_value"] == True, v["arg3_value"] == False))), (Or(v["arg4_value"] == True, v["arg4_value"] == False))), (Or(v["arg5_value"] == True, v["arg5_value"] == False))), (Or(v["arg6_value"] == True, v["arg6_value"] == False))), (Or(v["arg7_value"] == True, v["arg7_value"] == False))), (Or(v["arg8_value"] == True, v["arg8_value"] == False))), (Or(v["arg9_value"] == True, v["arg9_value"] == False))), (Or(v["arg10_value"] == True, v["arg10_value"] == False))), (Or(v["arg11_value"] == True, v["arg11_value"] == False))), (Or(v["arg12_value"] == True, v["arg12_value"] == False))), (Or(v["arg13_value"] == True, v["arg13_value"] == False))), (Or(v["arg14_value"] == True, v["arg14_value"] == False))))
)

def rule_67_func(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))
    arg7 = next(iter(arg7.values()))
    arg8 = next(iter(arg8.values()))
    arg9 = next(iter(arg9.values()))
    arg10 = next(iter(arg10.values()))
    arg11 = next(iter(arg11.values()))
    arg12 = next(iter(arg12.values()))
    arg13 = next(iter(arg13.values()))
    arg14 = next(iter(arg14.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, bool):
            return False
        if not isinstance(arg4, bool):
            return False
        if not isinstance(arg5, bool):
            return False
        if not isinstance(arg6, bool):
            return False
        if not isinstance(arg7, bool):
            return False
        if not isinstance(arg8, bool):
            return False
        if not isinstance(arg9, bool):
            return False
        if not isinstance(arg10, bool):
            return False
        if not isinstance(arg11, bool):
            return False
        if not isinstance(arg12, bool):
            return False
        if not isinstance(arg13, bool):
            return False
        if not isinstance(arg14, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_value = Bool('arg3_value')
        arg4_value = Bool('arg4_value')
        arg5_value = Bool('arg5_value')
        arg6_value = Bool('arg6_value')
        arg7_value = Bool('arg7_value')
        arg8_value = Bool('arg8_value')
        arg9_value = Bool('arg9_value')
        arg10_value = Bool('arg10_value')
        arg11_value = Bool('arg11_value')
        arg12_value = Bool('arg12_value')
        arg13_value = Bool('arg13_value')
        arg14_value = Bool('arg14_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == arg4)
        solver.add(arg5_value == arg5)
        solver.add(arg6_value == arg6)
        solver.add(arg7_value == arg7)
        solver.add(arg8_value == arg8)
        solver.add(arg9_value == arg9)
        solver.add(arg10_value == arg10)
        solver.add(arg11_value == arg11)
        solver.add(arg12_value == arg12)
        solver.add(arg13_value == arg13)
        solver.add(arg14_value == arg14)

        # Constraints for rule 67
        rule_67(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value, 'arg6_value': arg6_value, 'arg7_value': arg7_value, 'arg8_value': arg8_value, 'arg9_value': arg9_value, 'arg10_value': arg10_value, 'arg11_value': arg11_value, 'arg12_value': arg12_value, 'arg13_value': arg13_value, 'arg14_value': arg14_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_67(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value'], 'arg6_value': arg6['value'], 'arg7_value': arg7['value'], 'arg8_value': arg8['value'], 'arg9_value': arg9['value'], 'arg10_value': arg10['value'], 'arg11_value': arg11['value'], 'arg12_value': arg12['value'], 'arg13_value': arg13['value'], 'arg14_value': arg14['value']}, neg)
