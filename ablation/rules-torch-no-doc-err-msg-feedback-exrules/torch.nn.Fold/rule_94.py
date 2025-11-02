import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# When kernel_size, stride, dilation or padding is a tuple, the elements must have the same data type (Rule 94)

rule_94 = lambda s, v, n=False: (
    s.add(Not(Or([And(v_2 < (v["arg1_length"] + 1), True) for v_2 in range(6)])) if n else
          Or([And(v_2 < (v["arg1_length"] + 1), True) for v_2 in range(6)]))
)

def rule_94_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')

        # Value assignments
        solver.add(arg1_length == len(arg1))

        # Constraints for rule 94
        rule_94(solver, {'arg1_length': arg1_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_94(solver, {'arg1_length': arg1['length']}, neg)
