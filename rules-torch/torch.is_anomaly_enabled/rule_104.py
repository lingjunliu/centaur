import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If minimum value of a tensor is greater than 0 then maximum must be less than 100, or the tensor must be 1D, or the max must be a floating-point number, or minimum value is less than 10 (Rule 104)

rule_104 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_range"], 0) > 0, (Or(Or(Or(Or(Select(v["arg1_range"], 1) < 100, v["arg1_ndim"] == 1), v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), Select(v["arg1_range"], 0) < 10)), False)) if n else
          If(Select(v["arg1_range"], 0) > 0, (Or(Or(Or(Or(Select(v["arg1_range"], 1) < 100, v["arg1_ndim"] == 1), v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), Select(v["arg1_range"], 0) < 10)), False))
)

def rule_104_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 104
        rule_104(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_104(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype']}, neg)
