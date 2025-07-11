import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# file_buffer_size should be a power of 2 to potentially optimize memory allocation (Rule 34)

rule_34 = lambda s, v, n=False: (
    s.add(Not(Or([And(i < (20 + 1), And(v["arg1_value"] == 2, i)) for i in range(6)])) if n else
          Or([And(i < (20 + 1), And(v["arg1_value"] == 2, i)) for i in range(6)]))
)

def rule_34_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))

        # Constraints for rule 34
        rule_34(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_34(solver, {'arg1_value': arg1['value']}, neg)
