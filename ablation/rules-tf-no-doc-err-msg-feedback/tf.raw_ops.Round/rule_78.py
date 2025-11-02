import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Only allow the operation if tensor is of the right datatype and the list is of matching datatype (Rule 78)

rule_78 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_dtype"] == 3, v["arg1_length"] < 10, True)) if n else
          If(v["arg2_dtype"] == 3, v["arg1_length"] < 10, True))
)

def rule_78_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 78
        rule_78(solver, {'arg1_length': arg1_length, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_78(solver, {'arg1_length': arg1['length'], 'arg2_dtype': arg2['dtype']}, neg)
