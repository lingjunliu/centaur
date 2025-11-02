import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If length of tuple v_1 and v_2 must be same and minimum of v_3 should not be negative (Rule 99)

rule_99 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_length"] == v["arg2_length"], Select(v["arg3_range"], 0) >= 0)) if n else
          And(v["arg1_length"] == v["arg2_length"], Select(v["arg3_range"], 0) >= 0))
)

def rule_99_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_length = Int('arg2_length')
        arg3_range = Array('arg3_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_length == len(arg2))
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 99
        rule_99(solver, {'arg1_length': arg1_length, 'arg2_length': arg2_length, 'arg3_range': arg3_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_99(solver, {'arg1_length': arg1['length'], 'arg2_length': arg2['length'], 'arg3_range': arg3['range']}, neg)
