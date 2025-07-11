import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# scientific and shortest cannot both be true (Rule 9)

rule_9 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"], If(v["arg2_value"], False, False), False)) if n else
          If(v["arg1_value"], If(v["arg2_value"], False, False), False))
)

def rule_9_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Bool('arg2_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)

        # Constraints for rule 9
        rule_9(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_9(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
