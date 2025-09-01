import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The Ultimate Constraint - Attempt 3 (Rule 96)

rule_96 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(And(And(And(And((v["arg1_ndim"] > 0), (v["arg3_value"] > 0)), (v["arg4_value"] > 0)), (v["arg5_value"] >= 0)), (v["arg1_ndim"] == v["arg2_ndim"])), (And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) == Select(v["arg2_shape"], i)) for i in range(6)]))), (v["arg2_dtype"] == 4)), (If((And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 6)), (And(1 <= v["arg2_dtype"], v["arg2_dtype"] <= 6)), If((And(7 <= v["arg1_dtype"], v["arg1_dtype"] <= 11)), (And(7 <= v["arg2_dtype"], v["arg2_dtype"] <= 11)), True)))), (Or((v["arg6_length"] == 1), (v["arg6_length"] == 3)))), (If(Or((v["arg6_length"] == 1), (v["arg6_length"] == 3)), (And([Implies(i < (v["arg6_length"] - 1 + 1), Select(v["arg6_values"], i) > 0) for i in range(6)])), True))), (If((v["arg6_length"] == 1), (((Select(v["arg1_shape"], v["arg1_ndim"] - 1) + 2 * v["arg5_value"] - v["arg3_value"]) / v["arg4_value"]) + 1 == Select(v["arg6_values"], 0)), True))), (Select(v["arg1_shape"], v["arg1_ndim"] - 1) + 2 * v["arg5_value"] >= v["arg3_value"])), (Select(v["arg1_shape"], v["arg1_ndim"] - 1) >= v["arg4_value"]))) if n else
          And(And(And(And(And(And(And(And(And(And(And(And((v["arg1_ndim"] > 0), (v["arg3_value"] > 0)), (v["arg4_value"] > 0)), (v["arg5_value"] >= 0)), (v["arg1_ndim"] == v["arg2_ndim"])), (And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) == Select(v["arg2_shape"], i)) for i in range(6)]))), (v["arg2_dtype"] == 4)), (If((And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 6)), (And(1 <= v["arg2_dtype"], v["arg2_dtype"] <= 6)), If((And(7 <= v["arg1_dtype"], v["arg1_dtype"] <= 11)), (And(7 <= v["arg2_dtype"], v["arg2_dtype"] <= 11)), True)))), (Or((v["arg6_length"] == 1), (v["arg6_length"] == 3)))), (If(Or((v["arg6_length"] == 1), (v["arg6_length"] == 3)), (And([Implies(i < (v["arg6_length"] - 1 + 1), Select(v["arg6_values"], i) > 0) for i in range(6)])), True))), (If((v["arg6_length"] == 1), (((Select(v["arg1_shape"], v["arg1_ndim"] - 1) + 2 * v["arg5_value"] - v["arg3_value"]) / v["arg4_value"]) + 1 == Select(v["arg6_values"], 0)), True))), (Select(v["arg1_shape"], v["arg1_ndim"] - 1) + 2 * v["arg5_value"] >= v["arg3_value"])), (Select(v["arg1_shape"], v["arg1_ndim"] - 1) >= v["arg4_value"])))
)

def rule_96_func(arg1, arg2, arg3, arg4, arg5, arg6, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False
        if not (isinstance(arg5, (int, np.integer)) and not isinstance(arg5, bool)):
            return False
        if not (isinstance(arg6, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg6)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')
        arg5_value = Int('arg5_value')
        arg6_length = Int('arg6_length')
        arg6_values = Array('arg6_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == int(arg4))
        solver.add(arg5_value == int(arg5))
        solver.add(arg6_length == len(arg6))
        for i in range(len(arg6)):
            arg6_values = Store(arg6_values, i, arg6[i])

        # Constraints for rule 96
        rule_96(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value, 'arg6_length': arg6_length, 'arg6_values': arg6_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_96(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value'], 'arg6_length': arg6['length'], 'arg6_values': arg6['values']}, neg)
