import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If second argument is a number and first argument is a tensor, then second argument should be representable by tensor's dtype (Rule 29)

rule_29 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 1, And(-128 <= v["arg2_value"], v["arg2_value"] <= 127), If(v["arg1_dtype"] == 2, And(-32768 <= v["arg2_value"], v["arg2_value"] <= 32767), If(v["arg1_dtype"] == 3, And(-2147483648 <= v["arg2_value"], v["arg2_value"] <= 2147483647), If(v["arg1_dtype"] == 4, And(-9223372036854775808 <= v["arg2_value"], v["arg2_value"] <= 9223372036854775807), If(v["arg1_dtype"] == 5, And(0 <= v["arg2_value"], v["arg2_value"] <= 255), If(v["arg1_dtype"] == 6, And(1.17549435E-38 <= v["arg2_value"], v["arg2_value"] <= 3.4028235E38), If(v["arg1_dtype"] == 7, And(2.2250738585072014E-308 <= v["arg2_value"], v["arg2_value"] <= 1.7976931348623157E308), False)))))))) if n else
          If(v["arg1_dtype"] == 1, And(-128 <= v["arg2_value"], v["arg2_value"] <= 127), If(v["arg1_dtype"] == 2, And(-32768 <= v["arg2_value"], v["arg2_value"] <= 32767), If(v["arg1_dtype"] == 3, And(-2147483648 <= v["arg2_value"], v["arg2_value"] <= 2147483647), If(v["arg1_dtype"] == 4, And(-9223372036854775808 <= v["arg2_value"], v["arg2_value"] <= 9223372036854775807), If(v["arg1_dtype"] == 5, And(0 <= v["arg2_value"], v["arg2_value"] <= 255), If(v["arg1_dtype"] == 6, And(1.17549435E-38 <= v["arg2_value"], v["arg2_value"] <= 3.4028235E38), If(v["arg1_dtype"] == 7, And(2.2250738585072014E-308 <= v["arg2_value"], v["arg2_value"] <= 1.7976931348623157E308), False))))))))
)

def rule_29_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 29
        rule_29(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_29(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
