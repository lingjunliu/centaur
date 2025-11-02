import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# tensor shape matches given tuple shape (Rule 2)

rule_2 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_ndim"] == v["arg2_length"], And([Implies(i < (v["arg2_length"] - 1 + 1), Select(v["arg1_shape"], i) == Select(v["arg2_values"], i)) for i in range(6)]))) if n else
          And(v["arg1_ndim"] == v["arg2_length"], And([Implies(i < (v["arg2_length"] - 1 + 1), Select(v["arg1_shape"], i) == Select(v["arg2_values"], i)) for i in range(6)])))
)

def rule_2_func(arg1, arg2, solver=None, neg=False):
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
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 2
        rule_2(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_values': arg2_values, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_2(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_values': arg2['values'], 'arg2_length': arg2['length']}, neg)
