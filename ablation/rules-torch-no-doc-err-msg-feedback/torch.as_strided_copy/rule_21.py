import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# size and stride tuples must be non empty together or empty together (Rule 21)

rule_21 = lambda s, v, n=False: (
    s.add(Not(Or((And(v["arg1_length"] > 0, v["arg2_length"] > 0)), (And(v["arg1_length"] == 0, v["arg2_length"] == 0)))) if n else
          Or((And(v["arg1_length"] > 0, v["arg2_length"] > 0)), (And(v["arg1_length"] == 0, v["arg2_length"] == 0))))
)

def rule_21_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_length = Int('arg2_length')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_length == len(arg2))

        # Constraints for rule 21
        rule_21(solver, {'arg1_length': arg1_length, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_21(solver, {'arg1_length': arg1['length'], 'arg2_length': arg2['length']}, neg)
