import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If tuple dimensions exists and input dimensions exist, both are equal length (Rule 85)

rule_85 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_ndim"] >= 0, v["arg2_length"] >= 0), v["arg3_length"] >= 0), (v["arg2_length"] == v["arg3_length"]), False)) if n else
          If(And(And(v["arg1_ndim"] >= 0, v["arg2_length"] >= 0), v["arg3_length"] >= 0), (v["arg2_length"] == v["arg3_length"]), False))
)

def rule_85_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_length = Int('arg2_length')
        arg3_length = Int('arg3_length')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_length == len(arg2))
        solver.add(arg3_length == len(arg3))

        # Constraints for rule 85
        rule_85(solver, {'arg1_ndim': arg1_ndim, 'arg2_length': arg2_length, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_85(solver, {'arg1_ndim': arg1['ndim'], 'arg2_length': arg2['length'], 'arg3_length': arg3['length']}, neg)
