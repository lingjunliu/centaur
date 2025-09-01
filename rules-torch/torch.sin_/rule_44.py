import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The tensor's dtype must be suitable for numerical computations that prevent type casting to the desired output type Char, and its values must be within the valid range to avoid underflow and overflow errors during the sin_ operation  (Rule 44)

rule_44 = lambda s, v, n=False: (
    s.add(Not(Or((And((v["arg1_dtype"] > 5), (v["arg1_dtype"] < 11))), (And(And(And((v["arg1_dtype"] > 0), (v["arg1_dtype"] < 6)), (Select(v["arg1_range"], 0) > -1.0e+8)), (Select(v["arg1_range"], 1) < 1.0e+8))))) if n else
          Or((And((v["arg1_dtype"] > 5), (v["arg1_dtype"] < 11))), (And(And(And((v["arg1_dtype"] > 0), (v["arg1_dtype"] < 6)), (Select(v["arg1_range"], 0) > -1.0e+8)), (Select(v["arg1_range"], 1) < 1.0e+8)))))
)

def rule_44_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 44
        rule_44(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_44(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range']}, neg)
