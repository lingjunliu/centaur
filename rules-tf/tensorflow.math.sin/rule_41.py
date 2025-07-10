import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Check if the input tensor x contains no NaN values (Rule 41)

rule_41 = lambda s, v, n=False: (
    s.add(Not(Or([And(i < (v["arg1_ndim"] - 1 + 1), And(Select(v["arg1_shape"], i) != 0, (Or(Select(v["arg1_range"], 0) == Select(v["arg1_range"], 0), Select(v["arg1_range"], 1) == Select(v["arg1_range"], 1))))) for i in range(6)])) if n else
          Or([And(i < (v["arg1_ndim"] - 1 + 1), And(Select(v["arg1_shape"], i) != 0, (Or(Select(v["arg1_range"], 0) == Select(v["arg1_range"], 0), Select(v["arg1_range"], 1) == Select(v["arg1_range"], 1))))) for i in range(6)]))
)

def rule_41_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 41
        rule_41(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_41(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape']}, neg)
