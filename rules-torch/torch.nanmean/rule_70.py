import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# Prevent duplicate dim error: Elements in dimension tuple are all different in the tensor (Rule 70)

rule_70 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (If(v["arg1_length"] > 1, v["arg1_length"] - 2, 0) + 1), And(Select(v["arg1_values"], i) + 1 < v["arg2_ndim"], Select(v["arg1_values"], i) < v["arg2_ndim"])) for i in range(6)])) if n else
          And([Implies(i < (If(v["arg1_length"] > 1, v["arg1_length"] - 2, 0) + 1), And(Select(v["arg1_values"], i) + 1 < v["arg2_ndim"], Select(v["arg1_values"], i) < v["arg2_ndim"])) for i in range(6)]))
)

def rule_70_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 70
        rule_70(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_70(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values'], 'arg2_ndim': arg2['ndim']}, neg)
