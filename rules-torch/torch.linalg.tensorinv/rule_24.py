import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The product of the first ind dimensions equals the product of the rest of the dimensions (Rule 24)

rule_24 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_value"] > 0, v["arg2_value"] <= v["arg1_ndim"]), (Or([And(k < (10000 + 1), And(And((k == 1), (And([Implies(i < (v["arg1_ndim"] - 1 + 1), If(i < v["arg2_value"], (Or([And(x < (100 + 1), And(x == Select(v["arg1_shape"], i), k == k * x)) for x in range(6)])), True)) for i in range(6)]))), Or([And(l < (10000 + 1), And(And((l == 1), (And([Implies(j < (v["arg1_ndim"] - 1 + 1), If(j >= v["arg2_value"], (Or([And(y < (100 + 1), And(y == Select(v["arg1_shape"], j), l == l * y)) for y in range(6)])), True)) for j in range(6)]))), (k == l))) for l in range(6)]))) for k in range(6)])), True)) if n else
          If(And(v["arg2_value"] > 0, v["arg2_value"] <= v["arg1_ndim"]), (Or([And(k < (10000 + 1), And(And((k == 1), (And([Implies(i < (v["arg1_ndim"] - 1 + 1), If(i < v["arg2_value"], (Or([And(x < (100 + 1), And(x == Select(v["arg1_shape"], i), k == k * x)) for x in range(6)])), True)) for i in range(6)]))), Or([And(l < (10000 + 1), And(And((l == 1), (And([Implies(j < (v["arg1_ndim"] - 1 + 1), If(j >= v["arg2_value"], (Or([And(y < (100 + 1), And(y == Select(v["arg1_shape"], j), l == l * y)) for y in range(6)])), True)) for j in range(6)]))), (k == l))) for l in range(6)]))) for k in range(6)])), True))
)

def rule_24_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 24
        rule_24(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_24(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)
