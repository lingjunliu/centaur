import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# image: 3-D tensor, uint8, quality between 0 and 100, dimensions must be appropriate (Rule 104)

rule_104 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(v["arg1_ndim"] == 3, v["arg1_dtype"] == 5), 0 <= v["arg2_value"]), v["arg2_value"] <= 100), Select(v["arg1_shape"], 0) > 64), Select(v["arg1_shape"], 1) > 64), (Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) > 4096)), If(And(Select(v["arg1_shape"], 0) > 0, Select(v["arg1_shape"], 1) > 0), And((Select(v["arg1_shape"], 0) * 1.0 / Select(v["arg1_shape"], 1)) < 5, (Select(v["arg1_shape"], 1) * 1.0 / Select(v["arg1_shape"], 0)) < 5), False))) if n else
          And(And(And(And(And(And(And(v["arg1_ndim"] == 3, v["arg1_dtype"] == 5), 0 <= v["arg2_value"]), v["arg2_value"] <= 100), Select(v["arg1_shape"], 0) > 64), Select(v["arg1_shape"], 1) > 64), (Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) > 4096)), If(And(Select(v["arg1_shape"], 0) > 0, Select(v["arg1_shape"], 1) > 0), And((Select(v["arg1_shape"], 0) * 1.0 / Select(v["arg1_shape"], 1)) < 5, (Select(v["arg1_shape"], 1) * 1.0 / Select(v["arg1_shape"], 0)) < 5), False)))
)

def rule_104_func(arg1, arg2, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 104
        rule_104(solver, {'arg1_shape': arg1_shape, 'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_104(solver, {'arg1_shape': arg1['shape'], 'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
