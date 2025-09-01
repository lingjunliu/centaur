import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Check number of channels in output tensor when channels = 0 based on different dtypes, more compact (Rule 71)

rule_71 = lambda s, v, n=False: (
    s.add(Not(If((v["arg2_value"] == 0), (If(v["arg3_ndim"] >= 3, (And(v["arg3_dtype"] == v["arg1_value"], (Or(Or(Select(v["arg3_shape"], v["arg3_ndim"] - 1) == 1, Select(v["arg3_shape"], v["arg3_ndim"] - 1) == 3), Select(v["arg3_shape"], v["arg3_ndim"] - 1) == 4)))), True)), True)) if n else
          If((v["arg2_value"] == 0), (If(v["arg3_ndim"] >= 3, (And(v["arg3_dtype"] == v["arg1_value"], (Or(Or(Select(v["arg3_shape"], v["arg3_ndim"] - 1) == 1, Select(v["arg3_shape"], v["arg3_ndim"] - 1) == 3), Select(v["arg3_shape"], v["arg3_ndim"] - 1) == 4)))), True)), True))
)

def rule_71_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 71
        rule_71(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_ndim': arg3_ndim, 'arg3_dtype': arg3_dtype, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_71(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_ndim': arg3['ndim'], 'arg3_dtype': arg3['dtype'], 'arg3_shape': arg3['shape']}, neg)
