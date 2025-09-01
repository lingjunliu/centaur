import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Check if the input tensor requires too much memory and the last dim is not super big, given kernel_size and stride; Also check the size is reasonable and avoid OOM. Also at least one dimension is bigger than 0, avoid empty input and kernel_size and stride should not be extremely large and avoid zero size. Number of channels cannot be zero and input length cannot be zero, must be a non-complex tensor (Rule 114)

rule_114 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(And(And(And(And(And(And(Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) * Select(v["arg1_shape"], v["arg1_ndim"] - 1) < 1000000000, Select(v["arg1_shape"], v["arg1_ndim"] - 1) < 10000), Select(v["arg1_shape"], v["arg1_ndim"] - 1) > 0), v["arg2_value"] > 0), v["arg3_value"] > 0), Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) < 100000), (And(And(Select(v["arg1_shape"], 0) > 0, Select(v["arg1_shape"], 1) > 0), Select(v["arg1_shape"], v["arg1_ndim"] - 1) > 0))), v["arg2_value"] < 5000), v["arg3_value"] < 5000), Select(v["arg1_shape"], 0) != 0), Select(v["arg1_shape"], 1) != 0), Select(v["arg1_shape"], 1) > 0), Select(v["arg1_shape"], v["arg1_ndim"] - 1) != 0), v["arg1_dtype"] != 10), v["arg1_dtype"] != 11)) if n else
          And(And(And(And(And(And(And(And(And(And(And(And(And(And(Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) * Select(v["arg1_shape"], v["arg1_ndim"] - 1) < 1000000000, Select(v["arg1_shape"], v["arg1_ndim"] - 1) < 10000), Select(v["arg1_shape"], v["arg1_ndim"] - 1) > 0), v["arg2_value"] > 0), v["arg3_value"] > 0), Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) < 100000), (And(And(Select(v["arg1_shape"], 0) > 0, Select(v["arg1_shape"], 1) > 0), Select(v["arg1_shape"], v["arg1_ndim"] - 1) > 0))), v["arg2_value"] < 5000), v["arg3_value"] < 5000), Select(v["arg1_shape"], 0) != 0), Select(v["arg1_shape"], 1) != 0), Select(v["arg1_shape"], 1) > 0), Select(v["arg1_shape"], v["arg1_ndim"] - 1) != 0), v["arg1_dtype"] != 10), v["arg1_dtype"] != 11))
)

def rule_114_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 114
        rule_114(solver, {'arg1_shape': arg1_shape, 'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_114(solver, {'arg1_shape': arg1['shape'], 'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
