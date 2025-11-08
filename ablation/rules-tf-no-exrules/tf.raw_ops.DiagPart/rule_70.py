import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the tensor's dtype is complex128 and the ndim is between 2 and 4 inclusive, then the difference between max and min must be less than 200 and max value must be greater than zero. Otherwise, the ndim must be equal to 1 (Rule 70)

rule_70 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 10, (Or((And(And(And(v["arg1_ndim"] >= 2, v["arg1_ndim"] <= 4), (Select(v["arg1_range"], 1) - Select(v["arg1_range"], 0)) < 200), Select(v["arg1_range"], 1) > 0)), v["arg1_ndim"] == 1)), True)) if n else
          If(v["arg1_dtype"] == 10, (Or((And(And(And(v["arg1_ndim"] >= 2, v["arg1_ndim"] <= 4), (Select(v["arg1_range"], 1) - Select(v["arg1_range"], 0)) < 200), Select(v["arg1_range"], 1) > 0)), v["arg1_ndim"] == 1)), True))
)

def rule_70_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 70
        rule_70(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_70(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range']}, neg)
