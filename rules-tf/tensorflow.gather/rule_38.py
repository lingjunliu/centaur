import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If params has a dimension of size 1, axis must not refer to that dimension (Rule 38)

rule_38 = lambda s, v, n=False: (
    s.add(Not(If(Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) == 1) for i in range(6)]), And([Implies(j < (v["arg1_ndim"] - 1 + 1), If(Select(v["arg1_shape"], j) == 1, v["arg2_value"] != j, False)) for j in range(6)]), False)) if n else
          If(Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) == 1) for i in range(6)]), And([Implies(j < (v["arg1_ndim"] - 1 + 1), If(Select(v["arg1_shape"], j) == 1, v["arg2_value"] != j, False)) for j in range(6)]), False))
)

def rule_38_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 38
        rule_38(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_38(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)
