import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Comprehensive parameter constraints: delta, seed, image (Rule 63)

rule_63 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(And(And(-1 <= v["arg1_value"], v["arg1_value"] <= 1), v["arg2_length"] == 2), Select(v["arg2_values"], 0) >= 0), Select(v["arg2_values"], 1) >= 0), Select(v["arg2_values"], 0) != Select(v["arg2_values"], 1)), v["arg3_ndim"] >= 2), (And([Implies(i < (v["arg3_ndim"] - 1 + 1), Select(v["arg3_shape"], i) > 0) for i in range(6)]))), Select(v["arg3_range"], 0) <= Select(v["arg3_range"], 1)), (And(6 <= v["arg3_dtype"], Or(Or(Or(v["arg3_dtype"] <= 8, v["arg3_dtype"] == 9), v["arg3_dtype"] == 10), v["arg3_dtype"] == 0)))), (If(v["arg3_ndim"] == 4, Select(v["arg3_shape"], 0) != Select(v["arg3_shape"], 1), True)))) if n else
          And(And(And(And(And(And(And(And(And(And(-1 <= v["arg1_value"], v["arg1_value"] <= 1), v["arg2_length"] == 2), Select(v["arg2_values"], 0) >= 0), Select(v["arg2_values"], 1) >= 0), Select(v["arg2_values"], 0) != Select(v["arg2_values"], 1)), v["arg3_ndim"] >= 2), (And([Implies(i < (v["arg3_ndim"] - 1 + 1), Select(v["arg3_shape"], i) > 0) for i in range(6)]))), Select(v["arg3_range"], 0) <= Select(v["arg3_range"], 1)), (And(6 <= v["arg3_dtype"], Or(Or(Or(v["arg3_dtype"] <= 8, v["arg3_dtype"] == 9), v["arg3_dtype"] == 10), v["arg3_dtype"] == 0)))), (If(v["arg3_ndim"] == 4, Select(v["arg3_shape"], 0) != Select(v["arg3_shape"], 1), True))))
)

def rule_63_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')
        arg3_range = Array('arg3_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 63
        rule_63(solver, {'arg1_value': arg1_value, 'arg2_values': arg2_values, 'arg2_length': arg2_length, 'arg3_ndim': arg3_ndim, 'arg3_range': arg3_range, 'arg3_shape': arg3_shape, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_63(solver, {'arg1_value': arg1['value'], 'arg2_values': arg2['values'], 'arg2_length': arg2['length'], 'arg3_ndim': arg3['ndim'], 'arg3_range': arg3['range'], 'arg3_shape': arg3['shape'], 'arg3_dtype': arg3['dtype']}, neg)
