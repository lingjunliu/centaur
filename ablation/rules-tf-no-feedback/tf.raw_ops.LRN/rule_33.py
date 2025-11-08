import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# bias, alpha and beta should have the same dtype as the input (Rule 33)

rule_33 = lambda s, v, n=False: (
    s.add(Not(Or(Or((And(And(And(And(And(And(v["arg1_dtype"] == 6, -65504 <= v["arg2_value"]), v["arg2_value"] <= 65504), -65504 <= v["arg3_value"]), v["arg3_value"] <= 65504), -65504 <= v["arg4_value"]), v["arg4_value"] <= 65504)), (And(And(And(And(And(And(v["arg1_dtype"] == 7, -3.4028235e+38 <= v["arg2_value"]), v["arg2_value"] <= 3.4028235e+38), -3.4028235e+38 <= v["arg3_value"]), v["arg3_value"] <= 3.4028235e+38), -3.4028235e+38 <= v["arg4_value"]), v["arg4_value"] <= 3.4028235e+38))), (And(And(And(And(And(And(v["arg1_dtype"] == 8, -1.7976931348623157e+308 <= v["arg2_value"]), v["arg2_value"] <= 1.7976931348623157e+308), -1.7976931348623157e+308 <= v["arg3_value"]), v["arg3_value"] <= 1.7976931348623157e+308), -1.7976931348623157e+308 <= v["arg4_value"]), v["arg4_value"] <= 1.7976931348623157e+308)))) if n else
          Or(Or((And(And(And(And(And(And(v["arg1_dtype"] == 6, -65504 <= v["arg2_value"]), v["arg2_value"] <= 65504), -65504 <= v["arg3_value"]), v["arg3_value"] <= 65504), -65504 <= v["arg4_value"]), v["arg4_value"] <= 65504)), (And(And(And(And(And(And(v["arg1_dtype"] == 7, -3.4028235e+38 <= v["arg2_value"]), v["arg2_value"] <= 3.4028235e+38), -3.4028235e+38 <= v["arg3_value"]), v["arg3_value"] <= 3.4028235e+38), -3.4028235e+38 <= v["arg4_value"]), v["arg4_value"] <= 3.4028235e+38))), (And(And(And(And(And(And(v["arg1_dtype"] == 8, -1.7976931348623157e+308 <= v["arg2_value"]), v["arg2_value"] <= 1.7976931348623157e+308), -1.7976931348623157e+308 <= v["arg3_value"]), v["arg3_value"] <= 1.7976931348623157e+308), -1.7976931348623157e+308 <= v["arg4_value"]), v["arg4_value"] <= 1.7976931348623157e+308))))
)

def rule_33_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False
        if not isinstance(arg4, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Real('arg2_value')
        arg3_value = Real('arg3_value')
        arg4_value = Real('arg4_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == arg4)

        # Constraints for rule 33
        rule_33(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_33(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
