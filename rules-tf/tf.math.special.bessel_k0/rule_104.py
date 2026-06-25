import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If name includes 1, the max must be over or equal to 1, if name includes 2, the max must be over or equal to 2, and vice versa up to 9 (Rule 104)

rule_104 = lambda s, v, n=False: (
    s.add(Not(If(Or([And(i < (9 + 1), v["arg2_value"] == i) for i in range(6)]), Select(v["arg1_range"], 1) >= i, True)) if n else
          If(Or([And(i < (9 + 1), v["arg2_value"] == i) for i in range(6)]), Select(v["arg1_range"], 1) >= i, True))
)

def rule_104_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))

        # Constraints for rule 104
        rule_104(solver, {'arg1_range': arg1_range, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_104(solver, {'arg1_range': arg1['range'], 'arg2_value': arg2['value']}, neg)
