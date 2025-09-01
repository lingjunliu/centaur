import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The sum of the contracted dimensions' sizes must match (Rule 24)

rule_24 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg3_length"] > 0, v["arg4_length"] > 0), v["arg3_length"] == v["arg4_length"]), (And([Implies(i < (v["arg3_length"] - 1 + 1), Select(v["arg1_shape"], Select(v["arg3_values"], i)) == Select(v["arg2_shape"], Select(v["arg4_values"], i))) for i in range(6)])))) if n else
          And(And(And(v["arg3_length"] > 0, v["arg4_length"] > 0), v["arg3_length"] == v["arg4_length"]), (And([Implies(i < (v["arg3_length"] - 1 + 1), Select(v["arg1_shape"], Select(v["arg3_values"], i)) == Select(v["arg2_shape"], Select(v["arg4_values"], i))) for i in range(6)]))))
)

def rule_24_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False
        if not (isinstance(arg4, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())
        arg4_length = Int('arg4_length')
        arg4_values = Array('arg4_values', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])
        solver.add(arg4_length == len(arg4))
        for i in range(len(arg4)):
            arg4_values = Store(arg4_values, i, arg4[i])

        # Constraints for rule 24
        rule_24(solver, {'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg3_values': arg3_values, 'arg3_length': arg3_length, 'arg4_values': arg4_values, 'arg4_length': arg4_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_24(solver, {'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg3_values': arg3['values'], 'arg3_length': arg3['length'], 'arg4_values': arg4['values'], 'arg4_length': arg4['length']}, neg)
