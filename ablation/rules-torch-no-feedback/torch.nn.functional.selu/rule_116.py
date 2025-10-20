import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If inplace is True, input tensor must not have a qint8, quint8 or quint4 data type (Rule 116)

rule_116 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == True, v["arg1_dtype"] < 6, True)) if n else
          If(v["arg2_value"] == True, v["arg1_dtype"] < 6, True))
)

def rule_116_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Bool('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)

        # Constraints for rule 116
        rule_116(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_116(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
