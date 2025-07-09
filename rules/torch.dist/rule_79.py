import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# input tensors must be of compatible type, and not bool, correct grammar (Rule 79)

rule_79 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg1_dtype"] == 0, v["arg2_dtype"] == 0), False, Or((And((And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8)), (And(6 <= v["arg2_dtype"], v["arg2_dtype"] <= 8)))), (And((And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5)), (And(1 <= v["arg2_dtype"], v["arg2_dtype"] <= 5))))))) if n else
          If(Or(v["arg1_dtype"] == 0, v["arg2_dtype"] == 0), False, Or((And((And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8)), (And(6 <= v["arg2_dtype"], v["arg2_dtype"] <= 8)))), (And((And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5)), (And(1 <= v["arg2_dtype"], v["arg2_dtype"] <= 5)))))))
)

def rule_79_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 79
        rule_79(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_79(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype']}, neg)
