import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The product of the first ind dimensions equals the product of the rest of the dimensions (Rule 23)

rule_23 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_value"] > 0, v["arg2_value"] <= v["arg1_ndim"]), (Or([And(m < (1000000 + 1), Or([And(n < (1000000 + 1), And(And((And([Implies(i < (v["arg2_value"] - 1 + 1), m >= Select(v["arg1_shape"], i)) for i in range(6)])), (And([Implies(j < (v["arg1_ndim"] - 1 + 1), n >= Select(v["arg1_shape"], j)) for j in range(6)]))), m == n)) for n in range(6)])) for m in range(6)])), True)) if n else
          If(And(v["arg2_value"] > 0, v["arg2_value"] <= v["arg1_ndim"]), (Or([And(m < (1000000 + 1), Or([And(n < (1000000 + 1), And(And((And([Implies(i < (v["arg2_value"] - 1 + 1), m >= Select(v["arg1_shape"], i)) for i in range(6)])), (And([Implies(j < (v["arg1_ndim"] - 1 + 1), n >= Select(v["arg1_shape"], j)) for j in range(6)]))), m == n)) for n in range(6)])) for m in range(6)])), True))
)

def rule_23_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 23
        rule_23(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_23(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
