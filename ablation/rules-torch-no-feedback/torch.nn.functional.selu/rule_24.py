import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If inplace is true, then input tensor must be of floating or complex dtype. (Rule 24)

rule_24 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == True, Or((And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8)), (And(9 <= v["arg1_dtype"], v["arg1_dtype"] <= 10))), True)) if n else
          If(v["arg2_value"] == True, Or((And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8)), (And(9 <= v["arg1_dtype"], v["arg1_dtype"] <= 10))), True))
)

def rule_24_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 24
        rule_24(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_24(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
