import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# If first argument is a number and second argument is a tensor, then first argument should be representable by tensor's dtype (Rule 28)

rule_28 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_dtype"] == 1, And(-128 <= v["arg1_value"], v["arg1_value"] <= 127), If(v["arg2_dtype"] == 2, And(-32768 <= v["arg1_value"], v["arg1_value"] <= 32767), If(v["arg2_dtype"] == 3, And(-2147483648 <= v["arg1_value"], v["arg1_value"] <= 2147483647), If(v["arg2_dtype"] == 4, And(-9223372036854775808 <= v["arg1_value"], v["arg1_value"] <= 9223372036854775807), If(v["arg2_dtype"] == 5, And(0 <= v["arg1_value"], v["arg1_value"] <= 255), If(v["arg2_dtype"] == 6, And(1.17549435E-38 <= v["arg1_value"], v["arg1_value"] <= 3.4028235E38), If(v["arg2_dtype"] == 7, And(2.2250738585072014E-308 <= v["arg1_value"], v["arg1_value"] <= 1.7976931348623157E308), False)))))))) if n else
          If(v["arg2_dtype"] == 1, And(-128 <= v["arg1_value"], v["arg1_value"] <= 127), If(v["arg2_dtype"] == 2, And(-32768 <= v["arg1_value"], v["arg1_value"] <= 32767), If(v["arg2_dtype"] == 3, And(-2147483648 <= v["arg1_value"], v["arg1_value"] <= 2147483647), If(v["arg2_dtype"] == 4, And(-9223372036854775808 <= v["arg1_value"], v["arg1_value"] <= 9223372036854775807), If(v["arg2_dtype"] == 5, And(0 <= v["arg1_value"], v["arg1_value"] <= 255), If(v["arg2_dtype"] == 6, And(1.17549435E-38 <= v["arg1_value"], v["arg1_value"] <= 3.4028235E38), If(v["arg2_dtype"] == 7, And(2.2250738585072014E-308 <= v["arg1_value"], v["arg1_value"] <= 1.7976931348623157E308), False))))))))
)

def rule_28_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 28
        rule_28(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_28(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype']}, neg)
