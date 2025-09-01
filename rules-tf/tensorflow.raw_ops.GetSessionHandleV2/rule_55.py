import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The total size of the tensor (bytes (Rule 55)

rule_55 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 0, And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) < 100000) for i in range(6)]), If(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 5), And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) < 100000) for i in range(6)]), If(v["arg1_dtype"] == 2, And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) < 50000) for i in range(6)]), If(v["arg1_dtype"] == 3, And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) < 25000) for i in range(6)]), If(v["arg1_dtype"] == 4, And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) < 12500) for i in range(6)]), If(v["arg1_dtype"] == 6, And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) < 50000) for i in range(6)]), If(v["arg1_dtype"] == 7, And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) < 25000) for i in range(6)]), If(v["arg1_dtype"] == 8, And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) < 12500) for i in range(6)]), If(v["arg1_dtype"] == 9, And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) < 25000) for i in range(6)]), If(v["arg1_dtype"] == 10, And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) < 12500) for i in range(6)]), And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) < 100000) for i in range(6)])))))))))))) if n else
          If(v["arg1_dtype"] == 0, And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) < 100000) for i in range(6)]), If(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 5), And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) < 100000) for i in range(6)]), If(v["arg1_dtype"] == 2, And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) < 50000) for i in range(6)]), If(v["arg1_dtype"] == 3, And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) < 25000) for i in range(6)]), If(v["arg1_dtype"] == 4, And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) < 12500) for i in range(6)]), If(v["arg1_dtype"] == 6, And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) < 50000) for i in range(6)]), If(v["arg1_dtype"] == 7, And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) < 25000) for i in range(6)]), If(v["arg1_dtype"] == 8, And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) < 12500) for i in range(6)]), If(v["arg1_dtype"] == 9, And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) < 25000) for i in range(6)]), If(v["arg1_dtype"] == 10, And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) < 12500) for i in range(6)]), And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) < 100000) for i in range(6)]))))))))))))
)

def rule_55_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 55
        rule_55(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_55(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype']}, neg)
