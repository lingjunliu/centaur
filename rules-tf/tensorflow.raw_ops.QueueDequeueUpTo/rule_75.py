import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If timeout_ms == -1 and the queue is empty then n cannot be greater than the queues's max size if it has one (Rule 75)

rule_75 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == -1, True, True)) if n else
          If(v["arg2_value"] == -1, True, True))
)

def rule_75_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 75
        rule_75(solver, {'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_75(solver, {'arg2_value': arg2['value']}, neg)
