import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Tuple value must be lower the tensor max value if max bigger the 0 (Rule 116)

rule_116 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg2_range"], 1) > 0, (Or([And(i < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) < Select(v["arg2_range"], 1)) for i in range(6)])), True)) if n else
          If(Select(v["arg2_range"], 1) > 0, (Or([And(i < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) < Select(v["arg2_range"], 1)) for i in range(6)])), True))
)

def rule_116_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 116
        rule_116(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_116(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length'], 'arg2_range': arg2['range']}, neg)
