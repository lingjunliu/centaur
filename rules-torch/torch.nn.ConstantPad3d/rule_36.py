import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If dimension size is very small, each tuple padding value must be smaller to avoid overflow or negative size (Rule 36)

rule_36 = lambda s, v, n=False: (
    s.add(Not(If(Or([And(i < (2 + 1), Select(v["arg1_shape"], i + v["arg1_ndim"] - 3) < 5) for i in range(6)]), And([Implies(j < (5 + 1), Select(v["arg2_values"], j) < 2) for j in range(6)]), True)) if n else
          If(Or([And(i < (2 + 1), Select(v["arg1_shape"], i + v["arg1_ndim"] - 3) < 5) for i in range(6)]), And([Implies(j < (5 + 1), Select(v["arg2_values"], j) < 2) for j in range(6)]), True))
)

def rule_36_func(arg1, arg2, solver=None, neg=False):
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
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 36
        rule_36(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_36(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_values': arg2['values']}, neg)
