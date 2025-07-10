import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Reduction should not be constant and tensors should be of floating point type if margin is positive and P > 0 and swap is boolean (Rule 109)

rule_109 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And(And(And(And(v["arg1_value"] != 20, (Or(Or(v["arg2_dtype"] == 6, v["arg2_dtype"] == 7), v["arg2_dtype"] == 8))), v["arg2_dtype"] == v["arg3_dtype"]), v["arg2_dtype"] == v["arg4_dtype"]), v["arg5_value"] > 0), v["arg6_value"] > 0), (Or(v["arg7_value"] == True, v["arg7_value"] == False))), True, False)) if n else
          If(And(And(And(And(And(And(v["arg1_value"] != 20, (Or(Or(v["arg2_dtype"] == 6, v["arg2_dtype"] == 7), v["arg2_dtype"] == 8))), v["arg2_dtype"] == v["arg3_dtype"]), v["arg2_dtype"] == v["arg4_dtype"]), v["arg5_value"] > 0), v["arg6_value"] > 0), (Or(v["arg7_value"] == True, v["arg7_value"] == False))), True, False))
)

def rule_109_func(arg1, arg2, arg3, arg4, arg5, arg6, arg7, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))
    arg7 = next(iter(arg7.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False
        if not isinstance(arg5, (float, np.floating)):
            return False
        if not (isinstance(arg6, (int, np.integer)) and not isinstance(arg6, bool)):
            return False
        if not isinstance(arg7, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_dtype = Int('arg2_dtype')
        arg3_dtype = Int('arg3_dtype')
        arg4_dtype = Int('arg4_dtype')
        arg5_value = Real('arg5_value')
        arg6_value = Int('arg6_value')
        arg7_value = Bool('arg7_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values.index(arg1))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))
        solver.add(arg5_value == arg5)
        solver.add(arg6_value == int(arg6))
        solver.add(arg7_value == arg7)

        # Constraints for rule 109
        rule_109(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg3_dtype': arg3_dtype, 'arg4_dtype': arg4_dtype, 'arg5_value': arg5_value, 'arg6_value': arg6_value, 'arg7_value': arg7_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_109(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg3_dtype': arg3['dtype'], 'arg4_dtype': arg4['dtype'], 'arg5_value': arg5['value'], 'arg6_value': arg6['value'], 'arg7_value': arg7['value']}, neg)
