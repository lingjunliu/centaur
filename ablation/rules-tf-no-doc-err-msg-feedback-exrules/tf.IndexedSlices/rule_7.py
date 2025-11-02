import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Check that the indices tensor contains valid indices within the dense shape (Rule 7)

rule_7 = lambda s, v, n=False: (
    s.add(Not((And([Implies(i < (v["arg1_ndim"] - 1 + 1), And((Select(v["arg1_range"], 0) >= 0), (Select(v["arg1_range"], 1) < Select(v["arg2_values"], i)))) for i in range(6)]))) if n else
          (And([Implies(i < (v["arg1_ndim"] - 1 + 1), And((Select(v["arg1_range"], 0) >= 0), (Select(v["arg1_range"], 1) < Select(v["arg2_values"], i)))) for i in range(6)])))
)

def rule_7_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 7
        rule_7(solver, {'arg1_ndim': arg1_ndim, 'arg1_range': arg1_range, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_7(solver, {'arg1_ndim': arg1['ndim'], 'arg1_range': arg1['range'], 'arg2_values': arg2['values']}, neg)
