import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If both input and target are scalar tensors, then their underlying numerical values should also be scalars. (Rule 53)

rule_53 = lambda s, v, n=False: (
    s.add(Not(If(And((v["arg1_ndim"] == 0), (v["arg2_ndim"] == 0)), And((Or([And(a < (Select(v["arg1_range"], 1) + 1), True) for a in range(6)])), (Or([And(b < (Select(v["arg2_range"], 1) + 1), True) for b in range(6)]))), True)) if n else
          If(And((v["arg1_ndim"] == 0), (v["arg2_ndim"] == 0)), And((Or([And(a < (Select(v["arg1_range"], 1) + 1), True) for a in range(6)])), (Or([And(b < (Select(v["arg2_range"], 1) + 1), True) for b in range(6)]))), True))
)

def rule_53_func(arg1, arg2, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_ndim == arg2.ndim)
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 53
        rule_53(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg2_range': arg2_range, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_53(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg2_range': arg2['range'], 'arg2_ndim': arg2['ndim']}, neg)
