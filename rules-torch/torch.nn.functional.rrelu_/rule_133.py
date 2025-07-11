import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

#  If tensor doesn't consist of boolean values then, the boolean parameter must not be string type.  (Rule 133)

rule_133 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] != 0, Or(v["arg2_value"] == True, v["arg2_value"] == False), False)) if n else
          If(v["arg1_dtype"] != 0, Or(v["arg2_value"] == True, v["arg2_value"] == False), False))
)

def rule_133_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 133
        rule_133(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_133(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
