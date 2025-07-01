import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If v_1 is a list of integers and v_2 is a tensor, then if dtype of v_2 is one of complex64 or complex128, the length of the list must be equal to 2. (Rule 1413)

rule_1413 = lambda s, v, n=False: (
    s.add(Not(If((Or(v["arg2_dtype"] == 9, v["arg2_dtype"] == 10)), v["arg1_length"] == 2, False)) if n else
          If((Or(v["arg2_dtype"] == 9, v["arg2_dtype"] == 10)), v["arg1_length"] == 2, False))
)

def rule_1413_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 1413
        rule_1413(solver, {'arg1_length': arg1_length, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_1413(solver, {'arg1_length': arg1['length'], 'arg2_dtype': arg2['dtype']}, neg)
