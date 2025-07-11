import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Margin and P and swap are non negative only if all tensors have same floating point data type and reduction is not constant. (Rule 80)

rule_80 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And((Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8)), v["arg1_dtype"] == v["arg2_dtype"]), v["arg1_dtype"] == v["arg3_dtype"]), v["arg7_value"] != 20), And(And(v["arg4_value"] >= 0, v["arg5_value"] >= 0), (Or(v["arg6_value"] == True, v["arg6_value"] == False))), False)) if n else
          If(And(And(And((Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8)), v["arg1_dtype"] == v["arg2_dtype"]), v["arg1_dtype"] == v["arg3_dtype"]), v["arg7_value"] != 20), And(And(v["arg4_value"] >= 0, v["arg5_value"] >= 0), (Or(v["arg6_value"] == True, v["arg6_value"] == False))), False))
)

def rule_80_func(arg1, arg2, arg3, arg4, arg5, arg6, arg7, solver=None, neg=False):
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
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, (float, np.floating)):
            return False
        if not (isinstance(arg5, (int, np.integer)) and not isinstance(arg5, bool)):
            return False
        if not isinstance(arg6, bool):
            return False
        if not isinstance(arg7, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')
        arg3_dtype = Int('arg3_dtype')
        arg4_value = Real('arg4_value')
        arg5_value = Int('arg5_value')
        arg6_value = Bool('arg6_value')
        arg7_value = String('arg7_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_value == arg4)
        solver.add(arg5_value == int(arg5))
        solver.add(arg6_value == arg6)
        solver.add(arg7_value == list_of_string_values_torch.index(arg7))

        # Constraints for rule 80
        rule_80(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg3_dtype': arg3_dtype, 'arg4_value': arg4_value, 'arg5_value': arg5_value, 'arg6_value': arg6_value, 'arg7_value': arg7_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_80(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg3_dtype': arg3['dtype'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value'], 'arg6_value': arg6['value'], 'arg7_value': arg7['value']}, neg)
