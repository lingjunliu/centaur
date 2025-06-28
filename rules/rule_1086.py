import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# When specifying padding mode as constant, the constant value needs to be within the dtype range of the input data. (Rule 1086)

rule_1086 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == 10, (If(v["arg1_dtype"] == 1, (And(-128 <= v["arg3_value"], v["arg3_value"] <= 127)), If(v["arg1_dtype"] == 2, (And(-32768 <= v["arg3_value"], v["arg3_value"] <= 32767)), If(v["arg1_dtype"] == 3, (And(-2147483648 <= v["arg3_value"], v["arg3_value"] <= 2147483647)), If(v["arg1_dtype"] == 4, (And(-9223372036854775808 <= v["arg3_value"], v["arg3_value"] <= 9223372036854775807)), False))))), False)) if n else
          If(v["arg2_value"] == 10, (If(v["arg1_dtype"] == 1, (And(-128 <= v["arg3_value"], v["arg3_value"] <= 127)), If(v["arg1_dtype"] == 2, (And(-32768 <= v["arg3_value"], v["arg3_value"] <= 32767)), If(v["arg1_dtype"] == 3, (And(-2147483648 <= v["arg3_value"], v["arg3_value"] <= 2147483647)), If(v["arg1_dtype"] == 4, (And(-9223372036854775808 <= v["arg3_value"], v["arg3_value"] <= 9223372036854775807)), False))))), False))
)

def rule_1086_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False
        if not ((isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)) or isinstance(arg3, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == list_of_string_values.index(arg2))

        # Constraints for rule 1086
        rule_1086(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_1086(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
