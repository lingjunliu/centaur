import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the tensor is a scalar, then exclusive must be false, and reverse must be false (Rule 41)

rule_41 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 0, And(v["arg2_value"] == False, v["arg3_value"] == False), False)) if n else
          If(v["arg1_ndim"] == 0, And(v["arg2_value"] == False, v["arg3_value"] == False), False))
)

def rule_41_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = Bool('arg2_value')
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)

        # Constraints for rule 41
        rule_41(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_41(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
