import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If tensor's min is less than -10 and dimension is greater than 0, then dtype should not be bool or uint8 or int8. (Rule 138)

rule_138 = lambda s, v, n=False: (
    s.add(Not(If(And(Select(v["arg1_range"], 0) < -10, v["arg1_ndim"] > 0), And(And(v["arg1_dtype"] != 0, v["arg1_dtype"] != 1), v["arg1_dtype"] != 5), False)) if n else
          If(And(Select(v["arg1_range"], 0) < -10, v["arg1_ndim"] > 0), And(And(v["arg1_dtype"] != 0, v["arg1_dtype"] != 1), v["arg1_dtype"] != 5), False))
)

def rule_138_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 138
        rule_138(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_138(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range']}, neg)
