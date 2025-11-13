import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Frame_step constraint on upper bound with safety against int32 && padding (Rule 69)

rule_69 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg2_value"] > 0, v["arg2_value"] <= Select(v["arg1_shape"], v["arg1_ndim"] - 1)), (If(v["arg1_dtype"] == 3, Select(v["arg1_shape"], v["arg1_ndim"] - 1) < 2048, True))), Select(v["arg1_shape"], v["arg1_ndim"] - 1) - v["arg2_value"] > -28)) if n else
          And(And(And(v["arg2_value"] > 0, v["arg2_value"] <= Select(v["arg1_shape"], v["arg1_ndim"] - 1)), (If(v["arg1_dtype"] == 3, Select(v["arg1_shape"], v["arg1_ndim"] - 1) < 2048, True))), Select(v["arg1_shape"], v["arg1_ndim"] - 1) - v["arg2_value"] > -28))
)

def rule_69_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 69
        rule_69(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_69(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)
