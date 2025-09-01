import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If order has elements then dtype must be a floating type (Rule 91)

rule_91 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_length"] > 0, (And(6 <= v["arg2_dtype"], v["arg2_dtype"] <= 8)), True)) if n else
          If(v["arg1_length"] > 0, (And(6 <= v["arg2_dtype"], v["arg2_dtype"] <= 8)), True))
)

def rule_91_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all(isinstance(e, str) for e in arg1)):
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

        # Constraints for rule 91
        rule_91(solver, {'arg1_length': arg1_length, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_91(solver, {'arg1_length': arg1['length'], 'arg2_dtype': arg2['dtype']}, neg)
