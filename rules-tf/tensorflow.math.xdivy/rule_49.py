import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If y is scalar zero, then x must be scalar zero, and dtypes of both tensors must be valid float/complex (Rule 49)

rule_49 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_ndim"] == 0, (And(Select(v["arg2_range"], 0) == 0, Select(v["arg2_range"], 1) == 0))), (And(And(And(And(v["arg1_ndim"] == 0, Select(v["arg1_range"], 0) == 0), Select(v["arg1_range"], 1) == 0), (Or(Or(Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10))), (Or(Or(Or(Or(v["arg2_dtype"] == 6, v["arg2_dtype"] == 7), v["arg2_dtype"] == 8), v["arg2_dtype"] == 9), v["arg2_dtype"] == 10)))), False)) if n else
          If(And(v["arg2_ndim"] == 0, (And(Select(v["arg2_range"], 0) == 0, Select(v["arg2_range"], 1) == 0))), (And(And(And(And(v["arg1_ndim"] == 0, Select(v["arg1_range"], 0) == 0), Select(v["arg1_range"], 1) == 0), (Or(Or(Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10))), (Or(Or(Or(Or(v["arg2_dtype"] == 6, v["arg2_dtype"] == 7), v["arg2_dtype"] == 8), v["arg2_dtype"] == 9), v["arg2_dtype"] == 10)))), False))
)

def rule_49_func(arg1, arg2, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 49
        rule_49(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg2_range': arg2_range, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_49(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg2_range': arg2['range'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim']}, neg)
