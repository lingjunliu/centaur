import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if local_step is a power of 2, all dimensions of gradient tensor shape must have a value also equal to the power of 2 (Rule 87)

rule_87 = lambda s, v, n=False: (
    s.add(Not(If(Or([And(i < (v["arg1_value"] - 1 + 1), (And([Implies(k < (30 + 1), If((If((2 * k == v["arg1_value"]), (If((Or([And(j < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], j) == v["arg1_value"]) for j in range(6)])), True, False)), False)), True, False)) for k in range(6)]))) for i in range(6)]), True, False)) if n else
          If(Or([And(i < (v["arg1_value"] - 1 + 1), (And([Implies(k < (30 + 1), If((If((2 * k == v["arg1_value"]), (If((Or([And(j < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], j) == v["arg1_value"]) for j in range(6)])), True, False)), False)), True, False)) for k in range(6)]))) for i in range(6)]), True, False))
)

def rule_87_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 87
        rule_87(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_87(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim']}, neg)
