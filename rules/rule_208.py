import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if max - min <= 1 and tensor ndim greater than 2 and data type not equal to complex, then all dimensions of tensor have to be same (Rule 208)

rule_208 = lambda s, v, n=False: (
    s.add(Not(If(And(And((Select(v["arg1_range"], 1) - Select(v["arg1_range"], 0) <= 1), (v["arg1_ndim"] > 2)), (And(v["arg1_dtype"] != 9, v["arg1_dtype"] != 10))), And([Implies(i < (v["arg1_ndim"] - 2 + 1), Select(v["arg1_shape"], i) == Select(v["arg1_shape"], i + 1)) for i in range(6)]), False)) if n else
          If(And(And((Select(v["arg1_range"], 1) - Select(v["arg1_range"], 0) <= 1), (v["arg1_ndim"] > 2)), (And(v["arg1_dtype"] != 9, v["arg1_dtype"] != 10))), And([Implies(i < (v["arg1_ndim"] - 2 + 1), Select(v["arg1_shape"], i) == Select(v["arg1_shape"], i + 1)) for i in range(6)]), False))
)

def rule_208_func(arg1, solver=None, neg=False):
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
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 208
        rule_208(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_208(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg1_range': arg1['range']}, neg)
