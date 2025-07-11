import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If axis is not None and keepdims is False, and axis is a list, output rank is reduced by list's length (Rule 43)

rule_43 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg3_length"] > 0, v["arg2_value"] == False), v["arg1_ndim"] - v["arg3_length"] > 0, False)) if n else
          If(And(v["arg3_length"] > 0, v["arg2_value"] == False), v["arg1_ndim"] - v["arg3_length"] > 0, False))
)

def rule_43_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = Bool('arg2_value')
        arg3_length = Int('arg3_length')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == arg2)
        solver.add(arg3_length == len(arg3))

        # Constraints for rule 43
        rule_43(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_43(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_length': arg3['length']}, neg)
