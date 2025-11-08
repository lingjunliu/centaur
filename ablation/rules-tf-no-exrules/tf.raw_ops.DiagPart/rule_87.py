import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If dtype is half, then the tensor's ndim must be less than or equal to 3 and greater than 0 and max value is less than 100 and min value is greater than -50 and the tensor's max must be less than 50 and min value must be greater than -20 (Rule 87)

rule_87 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 2, (And(And(And(And(And(v["arg1_ndim"] <= 3, v["arg1_ndim"] > 0), Select(v["arg1_range"], 1) < 100), Select(v["arg1_range"], 0) > -50), Select(v["arg1_range"], 1) < 50), Select(v["arg1_range"], 0) > -20)), True)) if n else
          If(v["arg1_dtype"] == 2, (And(And(And(And(And(v["arg1_ndim"] <= 3, v["arg1_ndim"] > 0), Select(v["arg1_range"], 1) < 100), Select(v["arg1_range"], 0) > -50), Select(v["arg1_range"], 1) < 50), Select(v["arg1_range"], 0) > -20)), True))
)

def rule_87_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 87
        rule_87(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_87(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range']}, neg)
