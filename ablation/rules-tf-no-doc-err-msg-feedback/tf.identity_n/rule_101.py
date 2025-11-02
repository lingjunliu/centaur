import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If tensor dimension > 1 and dtype is int8 then each value need to be <= 127 and >= -128 (Rule 101)

rule_101 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_ndim"] > 1, v["arg1_dtype"] == 1), (And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(Select(v["arg1_range"], 1) <= 127, Select(v["arg1_range"], 0) >= -128)) for i in range(6)])), True)) if n else
          If(And(v["arg1_ndim"] > 1, v["arg1_dtype"] == 1), (And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(Select(v["arg1_range"], 1) <= 127, Select(v["arg1_range"], 0) >= -128)) for i in range(6)])), True))
)

def rule_101_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 101
        rule_101(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_101(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim']}, neg)
