import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the input tensor's dtype is complex, the lambd parameter must be a real number, i.e., its imaginary part should be zero. (Rule 18)

rule_18 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10), Or([And(x < (1000 + 1), v["arg2_value"] == x) for x in range(6)]), True)) if n else
          If(Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10), Or([And(x < (1000 + 1), v["arg2_value"] == x) for x in range(6)]), True))
)

def rule_18_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Real('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)

        # Constraints for rule 18
        rule_18(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_18(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
