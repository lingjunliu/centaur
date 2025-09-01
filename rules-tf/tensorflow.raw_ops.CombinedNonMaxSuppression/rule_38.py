import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# boxes values should be between 0 and 1 if clip_boxes is True (Rule 38)

rule_38 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"], And([Implies(i < (Select(v["arg1_shape"], 1) - 1 + 1), And([Implies(j < (3 + 1), And((Select(v["arg1_range"], 0) >= 0), (Select(v["arg1_range"], 1) <= 1))) for j in range(6)])) for i in range(6)]), True)) if n else
          If(v["arg2_value"], And([Implies(i < (Select(v["arg1_shape"], 1) - 1 + 1), And([Implies(j < (3 + 1), And((Select(v["arg1_range"], 0) >= 0), (Select(v["arg1_range"], 1) <= 1))) for j in range(6)])) for i in range(6)]), True))
)

def rule_38_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Bool('arg2_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == arg2)

        # Constraints for rule 38
        rule_38(solver, {'arg1_shape': arg1_shape, 'arg1_range': arg1_range, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_38(solver, {'arg1_shape': arg1['shape'], 'arg1_range': arg1['range'], 'arg2_value': arg2['value']}, neg)
