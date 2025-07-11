import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Structure is a tuple and its length equals the number of dimensions in the tensor and that length must be less than 5 and the sum is odd (Rule 73)

rule_73 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_ndim"] == v["arg2_length"], v["arg2_length"] < 5), (Or([And(i < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], i) % 2 != 0) for i in range(6)])))) if n else
          And(And(v["arg1_ndim"] == v["arg2_length"], v["arg2_length"] < 5), (Or([And(i < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], i) % 2 != 0) for i in range(6)]))))
)

def rule_73_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 73
        rule_73(solver, {'arg1_ndim': arg1_ndim, 'arg2_values': arg2_values, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_73(solver, {'arg1_ndim': arg1['ndim'], 'arg2_values': arg2['values'], 'arg2_length': arg2['length']}, neg)
