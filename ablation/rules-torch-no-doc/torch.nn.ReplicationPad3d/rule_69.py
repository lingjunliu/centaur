import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Valid dimension number, non-boolean type, all positive dimension sizes, valid padding, prevent padding related overflow/underflow (Rule 69)

rule_69 = lambda s, v, n=False: (
    s.add(Not(And(And((And((Or((v["arg1_ndim"] == 4), (v["arg1_ndim"] == 5))), (v["arg1_dtype"] != 0))), (And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)]))), (And(And(And(And(And((Select(v["arg1_shape"], 2) + (Select(v["arg2_values"], 0) + Select(v["arg2_values"], 1)) > 0), (Select(v["arg1_shape"], 3) + (Select(v["arg2_values"], 2) + Select(v["arg2_values"], 3)) > 0)), (Select(v["arg1_shape"], 4) + (Select(v["arg2_values"], 4) + Select(v["arg2_values"], 5)) > 0)), (Select(v["arg1_shape"], 2) + (Select(v["arg2_values"], 0) + Select(v["arg2_values"], 1)) < 2147483647)), (Select(v["arg1_shape"], 3) + (Select(v["arg2_values"], 2) + Select(v["arg2_values"], 3)) < 2147483647)), (Select(v["arg1_shape"], 4) + (Select(v["arg2_values"], 4) + Select(v["arg2_values"], 5)) < 2147483647))))) if n else
          And(And((And((Or((v["arg1_ndim"] == 4), (v["arg1_ndim"] == 5))), (v["arg1_dtype"] != 0))), (And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)]))), (And(And(And(And(And((Select(v["arg1_shape"], 2) + (Select(v["arg2_values"], 0) + Select(v["arg2_values"], 1)) > 0), (Select(v["arg1_shape"], 3) + (Select(v["arg2_values"], 2) + Select(v["arg2_values"], 3)) > 0)), (Select(v["arg1_shape"], 4) + (Select(v["arg2_values"], 4) + Select(v["arg2_values"], 5)) > 0)), (Select(v["arg1_shape"], 2) + (Select(v["arg2_values"], 0) + Select(v["arg2_values"], 1)) < 2147483647)), (Select(v["arg1_shape"], 3) + (Select(v["arg2_values"], 2) + Select(v["arg2_values"], 3)) < 2147483647)), (Select(v["arg1_shape"], 4) + (Select(v["arg2_values"], 4) + Select(v["arg2_values"], 5)) < 2147483647)))))
)

def rule_69_func(arg1, arg2, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 69
        rule_69(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_69(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_values': arg2['values']}, neg)
