import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# mean_ss and variance_ss should have compatible shapes for AddV2: either identical or broadcastable, counts must be scalar, and all three have consistent dtypes that can be handled by reciprocal (Rule 46)

rule_46 = lambda s, v, n=False: (
    s.add(Not(And(And(And(Or((And(v["arg1_ndim"] == v["arg2_ndim"], And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) == Select(v["arg2_shape"], i)) for i in range(6)]))), (And(And(v["arg1_ndim"] > 0, v["arg2_ndim"] > 0), And([Implies(i < (If(v["arg1_ndim"] >= v["arg2_ndim"], v["arg1_ndim"] - 1, v["arg2_ndim"] - 1) + 1), Or(Or(Or(Or((v["arg1_ndim"] - i - 1 < 0), (v["arg2_ndim"] - i - 1 < 0)), (Select(v["arg1_shape"], v["arg1_ndim"] - i - 1) == 1)), (Select(v["arg2_shape"], v["arg2_ndim"] - i - 1) == 1)), (Select(v["arg1_shape"], v["arg1_ndim"] - i - 1) == Select(v["arg2_shape"], v["arg2_ndim"] - i - 1)))) for i in range(6)])))), v["arg3_ndim"] == 0), (And(v["arg3_dtype"] == v["arg1_dtype"], v["arg1_dtype"] == v["arg2_dtype"]))), (Or(Or(Or(Or(Or(Or(Or(Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8), v["arg3_dtype"] == 3), v["arg3_dtype"] == 2), v["arg3_dtype"] == 1), v["arg3_dtype"] == 4), v["arg3_dtype"] == 5), v["arg3_dtype"] == 10), v["arg3_dtype"] == 11)))) if n else
          And(And(And(Or((And(v["arg1_ndim"] == v["arg2_ndim"], And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) == Select(v["arg2_shape"], i)) for i in range(6)]))), (And(And(v["arg1_ndim"] > 0, v["arg2_ndim"] > 0), And([Implies(i < (If(v["arg1_ndim"] >= v["arg2_ndim"], v["arg1_ndim"] - 1, v["arg2_ndim"] - 1) + 1), Or(Or(Or(Or((v["arg1_ndim"] - i - 1 < 0), (v["arg2_ndim"] - i - 1 < 0)), (Select(v["arg1_shape"], v["arg1_ndim"] - i - 1) == 1)), (Select(v["arg2_shape"], v["arg2_ndim"] - i - 1) == 1)), (Select(v["arg1_shape"], v["arg1_ndim"] - i - 1) == Select(v["arg2_shape"], v["arg2_ndim"] - i - 1)))) for i in range(6)])))), v["arg3_ndim"] == 0), (And(v["arg3_dtype"] == v["arg1_dtype"], v["arg1_dtype"] == v["arg2_dtype"]))), (Or(Or(Or(Or(Or(Or(Or(Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8), v["arg3_dtype"] == 3), v["arg3_dtype"] == 2), v["arg3_dtype"] == 1), v["arg3_dtype"] == 4), v["arg3_dtype"] == 5), v["arg3_dtype"] == 10), v["arg3_dtype"] == 11))))
)

def rule_46_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg3_ndim = Int('arg3_ndim')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 46
        rule_46(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype, 'arg3_ndim': arg3_ndim, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_46(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype'], 'arg3_ndim': arg3['ndim'], 'arg3_dtype': arg3['dtype']}, neg)
