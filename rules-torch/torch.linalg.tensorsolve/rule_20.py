import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Check the shape compatibility of A and B to prevent RuntimeError: Product of dimensions before B.ndim equals product of the rest, handle ndim(v1 (Rule 20)

rule_20 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] >= v["arg2_ndim"], Or([And(prod1 < (0 + 1), And((And(prod1 == 1, And([Implies(i < (If(v["arg2_ndim"] > 0, v["arg2_ndim"] - 1, 0) + 1), prod1 == prod1 * Select(v["arg1_shape"], i)) for i in range(6)]))), Or([And(prod2 < (0 + 1), And((And(prod2 == 1, And([Implies(i < (v["arg1_ndim"] - 1 + 1), prod2 == prod2 * Select(v["arg1_shape"], i)) for i in range(6)]))), prod1 == prod2)) for prod2 in range(6)]))) for prod1 in range(6)]), If(v["arg1_ndim"] < v["arg2_ndim"], Or([And(prod1 < (0 + 1), And((And(prod1 == 1, And([Implies(i < (v["arg1_ndim"] - 1 + 1), prod1 == prod1 * Select(v["arg1_shape"], i)) for i in range(6)]))), (prod1 == 1))) for prod1 in range(6)]), True))) if n else
          If(v["arg1_ndim"] >= v["arg2_ndim"], Or([And(prod1 < (0 + 1), And((And(prod1 == 1, And([Implies(i < (If(v["arg2_ndim"] > 0, v["arg2_ndim"] - 1, 0) + 1), prod1 == prod1 * Select(v["arg1_shape"], i)) for i in range(6)]))), Or([And(prod2 < (0 + 1), And((And(prod2 == 1, And([Implies(i < (v["arg1_ndim"] - 1 + 1), prod2 == prod2 * Select(v["arg1_shape"], i)) for i in range(6)]))), prod1 == prod2)) for prod2 in range(6)]))) for prod1 in range(6)]), If(v["arg1_ndim"] < v["arg2_ndim"], Or([And(prod1 < (0 + 1), And((And(prod1 == 1, And([Implies(i < (v["arg1_ndim"] - 1 + 1), prod1 == prod1 * Select(v["arg1_shape"], i)) for i in range(6)]))), (prod1 == 1))) for prod1 in range(6)]), True)))
)

def rule_20_func(arg1, arg2, solver=None, neg=False):
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
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 20
        rule_20(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_20(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim']}, neg)
