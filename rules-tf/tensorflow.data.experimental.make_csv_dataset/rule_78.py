import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If select_columns is specified, then number of elements in column_defaults has to match number of element in select_columns (Rule 78)

rule_78 = lambda s, v, n=False: (
    s.add(Not(v["arg1_length"] == v["arg2_length"]) if n else
          v["arg1_length"] == v["arg2_length"])
)

def rule_78_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all(isinstance(e, str) for e in arg1)):
            return False
        if not (isinstance(arg2, list) and all(isinstance(e, str) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_length = Int('arg2_length')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_length == len(arg2))

        # Constraints for rule 78
        rule_78(solver, {'arg1_length': arg1_length, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_78(solver, {'arg1_length': arg1['length'], 'arg2_length': arg2['length']}, neg)
