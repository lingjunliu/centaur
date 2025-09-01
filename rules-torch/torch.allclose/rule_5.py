import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If equal_nan is false, input and other should not have nan values at the same index (Rule 5)

rule_5 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] == False, And([Implies(i < (Select(v["arg1_shape"], 0) - 1 + 1), And([Implies(j < (Select(v["arg2_shape"], 0) - 1 + 1), (And(Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0), (Or(Select(v["arg1_range"], 0) != Select(v["arg1_range"], 0), Select(v["arg2_range"], 0) != Select(v["arg2_range"], 0)))))) for j in range(6)])) for i in range(6)]), True)) if n else
          If(v["arg3_value"] == False, And([Implies(i < (Select(v["arg1_shape"], 0) - 1 + 1), And([Implies(j < (Select(v["arg2_shape"], 0) - 1 + 1), (And(Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0), (Or(Select(v["arg1_range"], 0) != Select(v["arg1_range"], 0), Select(v["arg2_range"], 0) != Select(v["arg2_range"], 0)))))) for j in range(6)])) for i in range(6)]), True))
)

def rule_5_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_value = Bool('arg3_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_value == arg3)

        # Constraints for rule 5
        rule_5(solver, {'arg1_range': arg1_range, 'arg1_shape': arg1_shape, 'arg2_range': arg2_range, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_5(solver, {'arg1_range': arg1['range'], 'arg1_shape': arg1['shape'], 'arg2_range': arg2['range'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value']}, neg)
