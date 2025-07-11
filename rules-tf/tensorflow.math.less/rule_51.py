import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the tensor x has dimension 1, and the min value is greater than 5, tensor y should be float32 (Rule 51)

rule_51 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_ndim"] == 1, Select(v["arg1_range"], 0) > 5), v["arg2_dtype"] == 7, False)) if n else
          If(And(v["arg1_ndim"] == 1, Select(v["arg1_range"], 0) > 5), v["arg2_dtype"] == 7, False))
)

def rule_51_func(arg1, arg2, solver=None, neg=False):
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
        arg1_ndim = Int('arg1_ndim')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 51
        rule_51(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_51(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype']}, neg)
