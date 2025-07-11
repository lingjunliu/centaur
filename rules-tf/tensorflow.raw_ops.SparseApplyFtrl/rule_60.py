import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If Multiply Linear By Lr is enabled, the learning rate should be within a reasonable range (Rule 60)

rule_60 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"], (And(Select(v["arg2_range"], 0) >= 0.000001, Select(v["arg2_range"], 1) <= 1)), False)) if n else
          If(v["arg1_value"], (And(Select(v["arg2_range"], 0) >= 0.000001, Select(v["arg2_range"], 1) <= 1)), False))
)

def rule_60_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 60
        rule_60(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_60(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range']}, neg)
