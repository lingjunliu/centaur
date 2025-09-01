import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If enabled is true, the list must have length between 2 and 5. (Rule 114)

rule_114 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"], And(v["arg2_length"] >= 2, v["arg2_length"] <= 5), True)) if n else
          If(v["arg1_value"], And(v["arg2_length"] >= 2, v["arg2_length"] <= 5), True))
)

def rule_114_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_length = Int('arg2_length')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_length == len(arg2))

        # Constraints for rule 114
        rule_114(solver, {'arg1_value': arg1_value, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_114(solver, {'arg1_value': arg1['value'], 'arg2_length': arg2['length']}, neg)
