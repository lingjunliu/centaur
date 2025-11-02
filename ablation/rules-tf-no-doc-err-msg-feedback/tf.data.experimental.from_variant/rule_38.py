import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If v_1 is true then the maximum value of v_2 should be less than or equal to the length of v_3 (Rule 38)

rule_38 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"], Select(v["arg2_range"], 1) <= v["arg3_length"], True)) if n else
          If(v["arg1_value"], Select(v["arg2_range"], 1) <= v["arg3_length"], True))
)

def rule_38_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_length = Int('arg3_length')

        # Value assignments
        solver.add(arg1_value == arg1)
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_length == len(arg3))

        # Constraints for rule 38
        rule_38(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_38(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range'], 'arg3_length': arg3['length']}, neg)
