import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Input and output should have same data type, when not specified, they must have compatible datatypes, and small tensors should have small number of bins  (Rule 84)

rule_84 = lambda s, v, n=False: (
    s.add(Not(If((And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) < 100) for i in range(6)])), And((If(v["arg1_dtype"] < 6, (v["arg2_value"] < 1000), (v["arg2_value"] < 10000))), If(v["arg3_ndim"] > 0, v["arg1_dtype"] == v["arg3_dtype"], True)), True)) if n else
          If((And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) < 100) for i in range(6)])), And((If(v["arg1_dtype"] < 6, (v["arg2_value"] < 1000), (v["arg2_value"] < 10000))), If(v["arg3_ndim"] > 0, v["arg1_dtype"] == v["arg3_dtype"], True)), True))
)

def rule_84_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 84
        rule_84(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_84(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype'], 'arg3_ndim': arg3['ndim']}, neg)
