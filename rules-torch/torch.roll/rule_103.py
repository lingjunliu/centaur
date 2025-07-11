import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The dimensions of the input tensor must conform to valid ranges with existing dimensions. (Rule 103)

rule_103 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] >= 0, (And(-1 * v["arg1_ndim"] <= v["arg2_value"], v["arg2_value"] < v["arg1_ndim"])), (And(v["arg3_length"] > 0, (And([Implies(i < (v["arg3_length"] - 1 + 1), And((-1 * v["arg1_ndim"]) <= Select(v["arg3_values"], i), Select(v["arg3_values"], i) < v["arg1_ndim"])) for i in range(6)])))))) if n else
          If(v["arg2_value"] >= 0, (And(-1 * v["arg1_ndim"] <= v["arg2_value"], v["arg2_value"] < v["arg1_ndim"])), (And(v["arg3_length"] > 0, (And([Implies(i < (v["arg3_length"] - 1 + 1), And((-1 * v["arg1_ndim"]) <= Select(v["arg3_values"], i), Select(v["arg3_values"], i) < v["arg1_ndim"])) for i in range(6)]))))))
)

def rule_103_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = Int('arg2_value')
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 103
        rule_103(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_length': arg3_length, 'arg3_values': arg3_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_103(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_length': arg3['length'], 'arg3_values': arg3['values']}, neg)
