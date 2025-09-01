import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if x has more than 1 dimension and is not empty, than elements of v must support subtraction from elements of x, and type of elements of x,v, and i, must be numerical except boolean or string (Rule 111)

rule_111 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_ndim"] > 1, (And([Implies(j < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], j) > 0) for j in range(6)]))), And(And((And(v["arg1_dtype"] >= 1, v["arg1_dtype"] <= 10)), (And(v["arg2_dtype"] >= 1, v["arg2_dtype"] <= 10))), (And(v["arg3_dtype"] >= 1, v["arg3_dtype"] <= 10))), True)) if n else
          If(And(v["arg1_ndim"] > 1, (And([Implies(j < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], j) > 0) for j in range(6)]))), And(And((And(v["arg1_dtype"] >= 1, v["arg1_dtype"] <= 10)), (And(v["arg2_dtype"] >= 1, v["arg2_dtype"] <= 10))), (And(v["arg3_dtype"] >= 1, v["arg3_dtype"] <= 10))), True))
)

def rule_111_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 111
        rule_111(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_111(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg3_dtype': arg3['dtype']}, neg)
