import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# cross_trainer_cache should be a tuple (Rule 98)

rule_98 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(v["arg1_length"] == 0, v["arg2_length"] == 0), v["arg3_length"] == 0), v["arg4_length"] == 0)) if n else
          Or(Or(Or(v["arg1_length"] == 0, v["arg2_length"] == 0), v["arg3_length"] == 0), v["arg4_length"] == 0))
)

def rule_98_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all(isinstance(e, str) for e in arg1)):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, tuple) and all(isinstance(e, (float, np.floating)) for e in arg3)):
            return False
        if not (isinstance(arg4, tuple) and all(isinstance(e, bool) for e in arg4)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_length = Int('arg2_length')
        arg3_length = Int('arg3_length')
        arg4_length = Int('arg4_length')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_length == len(arg2))
        solver.add(arg3_length == len(arg3))
        solver.add(arg4_length == len(arg4))

        # Constraints for rule 98
        rule_98(solver, {'arg1_length': arg1_length, 'arg2_length': arg2_length, 'arg3_length': arg3_length, 'arg4_length': arg4_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_98(solver, {'arg1_length': arg1['length'], 'arg2_length': arg2['length'], 'arg3_length': arg3['length'], 'arg4_length': arg4['length']}, neg)
