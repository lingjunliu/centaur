import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The MOST complete validation - all requirements (Rule 41)

rule_41 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(v["arg1_value"] >= -1.0, v["arg1_value"] <= 1.0), v["arg2_ndim"] >= 3), Select(v["arg2_shape"], v["arg2_ndim"] - 1) == 3), (And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) > 0) for i in range(6)]))), v["arg2_dtype"] != 10), v["arg2_dtype"] != 11), (If(v["arg2_dtype"] == 1, (And(Select(v["arg2_range"], 0) >= -128, Select(v["arg2_range"], 1) <= 127)), If(v["arg2_dtype"] == 2, (And(Select(v["arg2_range"], 0) >= -32768, Select(v["arg2_range"], 1) <= 32767)), If(v["arg2_dtype"] == 3, (And(Select(v["arg2_range"], 0) >= -2147483648, Select(v["arg2_range"], 1) <= 2147483647)), If(v["arg2_dtype"] == 4, (And(Select(v["arg2_range"], 0) >= -9223372036854775808, Select(v["arg2_range"], 1) <= 9223372036854775807)), If(v["arg2_dtype"] == 5, (And(Select(v["arg2_range"], 0) >= 0, Select(v["arg2_range"], 1) <= 255)), (Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8)))))))))) if n else
          And(And(And(And(And(And(And(v["arg1_value"] >= -1.0, v["arg1_value"] <= 1.0), v["arg2_ndim"] >= 3), Select(v["arg2_shape"], v["arg2_ndim"] - 1) == 3), (And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) > 0) for i in range(6)]))), v["arg2_dtype"] != 10), v["arg2_dtype"] != 11), (If(v["arg2_dtype"] == 1, (And(Select(v["arg2_range"], 0) >= -128, Select(v["arg2_range"], 1) <= 127)), If(v["arg2_dtype"] == 2, (And(Select(v["arg2_range"], 0) >= -32768, Select(v["arg2_range"], 1) <= 32767)), If(v["arg2_dtype"] == 3, (And(Select(v["arg2_range"], 0) >= -2147483648, Select(v["arg2_range"], 1) <= 2147483647)), If(v["arg2_dtype"] == 4, (And(Select(v["arg2_range"], 0) >= -9223372036854775808, Select(v["arg2_range"], 1) <= 9223372036854775807)), If(v["arg2_dtype"] == 5, (And(Select(v["arg2_range"], 0) >= 0, Select(v["arg2_range"], 1) <= 255)), (Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8))))))))))
)

def rule_41_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 41
        rule_41(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype, 'arg2_shape': arg2_shape, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_41(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_shape': arg2['shape'], 'arg2_range': arg2['range']}, neg)
