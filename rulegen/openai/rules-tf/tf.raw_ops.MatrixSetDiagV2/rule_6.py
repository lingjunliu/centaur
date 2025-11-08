import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# tuple k: diagonal rank depends on band width (Rule 6)

rule_6 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_ndim"] >= 2, (If(Select(v["arg3_values"], 0) == Select(v["arg3_values"], 1), v["arg2_ndim"] == v["arg1_ndim"] - 1, v["arg2_ndim"] == v["arg1_ndim"])))) if n else
          And(v["arg1_ndim"] >= 2, (If(Select(v["arg3_values"], 0) == Select(v["arg3_values"], 1), v["arg2_ndim"] == v["arg1_ndim"] - 1, v["arg2_ndim"] == v["arg1_ndim"]))))
)

def rule_6_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_ndim = Int('arg2_ndim')
        arg3_values = Array('arg3_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 6
        rule_6(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg3_values': arg3_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_6(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg3_values': arg3['values']}, neg)
